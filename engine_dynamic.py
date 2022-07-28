import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from einops import rearrange

import torch.nn.functional as F
from timm.utils import accuracy, AverageMeter
from util import get_grad_norm, reduce_tensor
from utils.log_helper import default_logger as logger
from logger import create_logger
from utils.masking_generator import RandomMaskingGenerator
import torch.distributed as dist
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util import  get_grad_norm, auto_resume_helper, reduce_tensor, sim_dis_compute, loss_kl_neck_single, save_last_checkpoint

def train_one_epoch_dynamic(config, teacher, student, masked_autoencoder, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, T, tb_logger):
    student.train()
    teacher.eval()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hard_loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    loss_func = nn.MSELoss()
    
    N =  num_steps
    M =  num_steps
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    
    for idx, (samples_masks, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
        samples = samples_masks[0]

        # torch.Size([128, 3, 224, 224])     
        samples= samples.cuda(non_blocking=True)
        bool_masked_pos = samples_masks[1]
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.TRAIN.DISTILLATION.TYPE == 'logits':
            outputs_student = student(samples)
        else:
            feats_s, outputs_student = student(samples)            

        with torch.no_grad():
            if config.TRAIN.DISTILLATION.TYPE == 'logits':
                outputs_teacher = teacher(samples)
            else:
                feats_t, outputs_teacher = teacher(samples)
                
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            
            loss_feats = None
            # images.shape                                                        │  
            index = config.TRAIN.DISTILLATION.FEAT_INDEX    
            X  = feats_s[index].flatten(1)
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K, K, device) / num_steps                        
            images =  feats_s[index].permute(0,2,1)
            images = images.view(images.shape[0],images.shape[1],int(np.sqrt(images.shape[2])), int(np.sqrt(images.shape[2])))
            
            Y = feats_t[index].flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            
            hsic_matrix[idx, 1] += HSIC(K, L, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L, L, device) / num_steps
            
            # Dynamic Masked Ratio
            dynamic_ratio = 1.0 - torch.sigmoid(torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device))))) # the larger the similarity score, the smaller the teacher participant
            num_patches = bool_masked_pos.shape[1]
            num_mask = int(num_patches * dynamic_ratio)
            bool_masked_pos_dynamic = []
            for i in range(bool_masked_pos.shape[0]):
                mask = np.hstack([np.zeros(num_patches - num_mask),np.ones(num_mask)])
                np.random.shuffle(mask)
                mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                bool_masked_pos_dynamic.append(mask)
            bool_masked_pos = torch.stack(bool_masked_pos_dynamic)
            
            target_images = feats_t[index].permute(0,2,1)
            target_images = target_images.view(target_images.shape[0],target_images.shape[1], int(np.sqrt(target_images.shape[2])), int(np.sqrt(target_images.shape[2])))
            
            # original feature shape:::: B Ph*Pw C
            # feats_s[0].shape                                                    │  
            # torch.Size([128, 784, 96])      
            # feats_s[1].shape                                                     
            # torch.Size([128, 196, 192])                                                                 
            # feats_s[2].shape                                                    
            # torch.Size([128, 49, 384])                                                                  
            # feats_s[3].shape                                                    
            # torch.Size([128, 49, 384])                                                                  
            
            # Large
            # feats_t[0].shape                                                    
            # torch.Size([128, 784, 384])                                                                
            # feats_t[1].shape                                                    
            # torch.Size([128, 196, 768])                                                                                                                                               
            # feats_t[2].shape                                                                
            # torch.Size([128, 49, 1536])                                                                
            # feats_t[3].shape                                                    
            # torch.Size([128, 49, 1536])

            # Base
            # feats_t[0].shape                                                                                 
            # torch.Size([32, 784, 256])                                                                             
            # feats_t[1].shape                                                                                 
            # torch.Size([32, 196, 512])                                                                             
            # feats_t[2].shape                                                                                 
            # torch.Size([32, 49, 1024])                                                                             
            # feats_t[3].shape                                                                                 
            # torch.Size([32, 49, 1024])
            
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        # calculate the predict label
                        unnorm_images = target_images     
                        if normlize_target:
                            images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            # images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                            #     ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                            images_norm  = F.normalize(images_squeeze, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                        else:
                            images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                        B, _, C = images_patch.shape # torch.Size([128, 49, xxx])
                    
                    # import pdb; pdb.set_trace()                        
                    outputs = masked_autoencoder(images, target_images, bool_masked_pos)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    loss_feats = loss_func(input=outputs, target=images_patch)
                    # loss_feats = None
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + config.TRAIN.DISTILLATION.BETA * loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                        
                    
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            loss = loss_soft * 0.9 + loss_hard * 0.1
            optimizer.zero_grad()
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
        if loss_feats is not None:
            feat_loss_meter.update(loss_feats.item(), targets.size(0))        
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
                    f'loss_features {feat_loss_meter.val:.4f} ({feat_loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1

                # if dist.get_rank() == 0:
                #     tb_logger.add_scalar('lr', lr, curr_step)
                #     tb_logger.add_scalar('hard_loss_train', hard_loss_meter.avg, curr_step)
                #     tb_logger.add_scalar('soft_loss_train', soft_loss_meter.avg, curr_step)
                #     tb_logger.add_scalar('feat_loss_train', feat_loss_meter.avg, curr_step)
                #     tb_logger.add_scalar('total_loss_train', loss_meter.avg, curr_step)
    
            else:
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1
    
                # if dist.get_rank() == 0:
                #     tb_logger.add_scalar('lr', lr, curr_step)
                #     tb_logger.add_scalar('hard_loss_train', hard_loss_meter.avg, curr_step)
                #     tb_logger.add_scalar('soft_loss_train', soft_loss_meter.avg, curr_step)
                #     # tb_logger.add_scalar('feat_loss_train', feat_loss_meter.avg, curr_step)
                #     tb_logger.add_scalar('total_loss_train', loss_meter.avg, curr_step)
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix

    
@torch.no_grad()
def validate_dynamic(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    # for idx, (images, target) in enumerate(data_loader):
    #     images = images.cuda(non_blocking=True)
    #     target = target.cuda(non_blocking=True)        
    for idx, (samples_masks, target) in enumerate(data_loader):    
        images = samples_masks[0]
        images= images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # compute output
        if config.TRAIN.DISTILLATION.TYPE == 'logits' or config.TRAIN.DISTILLATION.TYPE =='':
            output = model(images)
        else: 
            feats, output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc3, acc5 = accuracy(output, target, topk=(1,3,5))

        acc1 = reduce_tensor(acc1)
        acc3 = reduce_tensor(acc3)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc3_meter.update(acc3.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@3 {acc3_meter.val:.3f} ({acc3_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@3 {acc3_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc3_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
