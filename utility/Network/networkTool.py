import glob
import math
import torch
import random
import numpy as np
import albumentations as A

from math import ceil
from os import makedirs
from os.path import exists, join
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import Sampler
from torch.utils.data import SubsetRandomSampler, DataLoader
# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Utility Functions
# --------------------------------------------------------------------------------
iteration = 0
iteration_val = 0

class AvgMeter:
    
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def cross_entropy(preds, targets, reduction='none'):
    
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# --------------------------------------------------------------------------------
# [LSV-Loc EVALUATION] Generate Embeddings
# --------------------------------------------------------------------------------

def lidar_image_embeddings(valid_loader, model):

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, position=0, leave=False):
            image_embeddings = model.module.lidar_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def camera_image_embeddings(valid_loader, model):

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, position=0, leave=False):
            image_embeddings = model.module.camera_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] T-InfoNCE Loss
# --------------------------------------------------------------------------------

def partial_detach_gather_utm(local_utm: torch.Tensor) -> torch.Tensor:
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gather_list = [torch.zeros_like(local_utm) for _ in range(world_size)]
    dist.all_gather(gather_list, local_utm)
    gather_list[rank] = local_utm

    global_utm = torch.cat(gather_list, dim=0)
    chunks = global_utm.chunk(world_size, dim=0)
    
    new_chunks = []
    for i, chunk in enumerate(chunks):
        if i == rank:
            new_chunks.append(chunk)          # 내 것: autograd 유지
        else:
            new_chunks.append(chunk.detach()) # 다른 랭크: detach
    partial_detached_global = torch.cat(new_chunks, dim=0)
    return partial_detached_global

def partial_detach_gather_embeddings(local_embeddings: torch.Tensor) -> torch.Tensor:

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gather_list = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
    dist.all_gather(gather_list, local_embeddings)
    gather_list[rank] = local_embeddings

    global_embeddings = torch.cat(gather_list, dim=0)

    chunks = global_embeddings.chunk(world_size, dim=0)

    new_chunks = []
    for i, chunk in enumerate(chunks):
        if i == rank:
            new_chunks.append(chunk)          # 내 것: Autograd 유지
        else:
            new_chunks.append(chunk.detach()) # 다른 랭크: detach

    partial_detached_global = torch.cat(new_chunks, dim=0)
    return partial_detached_global

def InfoNCE(lidar_image_embeddings: torch.Tensor, camera_image_embeddings: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 각 임베딩을 정규화
    image_features1 = F.normalize(lidar_image_embeddings, dim=-1)
    image_features2 = F.normalize(camera_image_embeddings, dim=-1)
    
    # 두 임베딩 간 내적 결과에 스케일을 곱해 logits를 계산합니다.
    logits_per_image1 = logit_scale * torch.matmul(image_features1, image_features2.T)
    logits_per_image2 = logits_per_image1.T
    
    # 배치 내 각 예제에 대해, 양성 쌍은 동일 인덱스이므로 labels는 0부터 (B-1)까지.
    labels = torch.arange(image_features1.shape[0], dtype=torch.long, device=image_features1.device)
    
    # 양쪽에 대해 loss를 구하고 평균
    loss = (loss_fn(logits_per_image1, labels) + loss_fn(logits_per_image2, labels)) / 2
    return loss

def contrastive_loss(lidar_image_embeddings: torch.Tensor,camera_image_embeddings: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
  
    # Calculating the Loss
    logits = (lidar_image_embeddings @ camera_image_embeddings.T) / logit_scale
    camera_similarity = camera_image_embeddings @ camera_image_embeddings.T
    lidar_similarity = lidar_image_embeddings @ lidar_image_embeddings.T
    targets = F.softmax(
        (camera_similarity + lidar_similarity) / 2 * logit_scale, dim=-1
    )
    lidar_loss = cross_entropy(logits, targets, reduction='none')
    camera_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (camera_loss + lidar_loss) / 2.0 # shape: (batch_size)
    loss = loss.mean()
    
    return loss

def train_step_BCL(model, batch, temperature: float = 0.07):
    
    local_embeddings_lidar = model.module.lidar_embeddings(batch)     # shape: (B, D)
    local_embeddings_camera = model.module.camera_embeddings(batch)    # shape: (B, D)

    global_embeddings_lidar  = partial_detach_gather_embeddings(local_embeddings_lidar)     # shape: (B * num_gpus, D)
    global_embeddings_camera = partial_detach_gather_embeddings(local_embeddings_camera)    # shape: (B * num_gpus, D)
    
    loss = InfoNCE(global_embeddings_lidar, global_embeddings_camera, model.module.logit_scale.exp())

    return loss

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] T-InfoNCE + EPro-PnP Loss
# --------------------------------------------------------------------------------

def train_step_PnP(model, batch, temperature: float = 0.07):
    
    local_embeddings_lidar = model.module.lidar_embeddings(batch)     # shape: (B, D)
    local_embeddings_camera = model.module.camera_embeddings(batch)    # shape: (B, D)

    global_embeddings_lidar  = partial_detach_gather_embeddings(local_embeddings_lidar)     # shape: (B * num_gpus, D)
    global_embeddings_camera = partial_detach_gather_embeddings(local_embeddings_camera)    # shape: (B * num_gpus, D)
    
    loss_InfoNCE = InfoNCE(global_embeddings_lidar, global_embeddings_camera, model.module.logit_scale.exp())

    
    

    return loss


# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Validation Functions
# --------------------------------------------------------------------------------

def valid_epoch(model, valid_loader, rank, writer, CONFIG):
    global iteration_val
    
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), desc="\033[92m===: [LSV-Loc TRAINING] VALID: Iteration \033[0m", position=0, leave=True, disable=rank != 0)
    for batch in tqdm_object:
        batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
        loss = model(batch)

        if rank == 0:
            writer.add_scalar("Loss/Valid-Batch", loss.item(), iteration_val)
            iteration_val = iteration_val+1
        
        count = batch["camera_image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def valid_recall(model, valid_dataset, valid_loader, rank, writer, epoch, CONFIG):
    
    # [LSV-Loc EVALUATION] Generate Embeddings
    print('===: [LSV-Loc EVALUATION] Embedding GEN: LiDAR')
    lidar_embeddings = lidar_image_embeddings(valid_loader, model)
    lidar_embeddings = lidar_embeddings.cuda()  

    print('===: [LSV-Loc EVALUATION] Embedding GEN: Camera')
    camera_embeddings = camera_image_embeddings(valid_loader, model)
    camera_embeddings = camera_embeddings.cuda()
    
    print('===: [LSV-Loc EVALUATION] Pose GEN')
    num_matches = 0
    total_queries = valid_dataset.dbStruct.numQ

    print('\033[92m===: [LSV-Loc EVALUATION] Evaluation START\033[0m')
    recall_Idx = [1,5,10,15,20]
    query_predict = []
    recall_results = {k: 0 for k in recall_Idx}

    for i, filename in tqdm(enumerate(valid_dataset.dbStruct.qImage)):
        max_k = max(recall_Idx)
        
        dot_similarity = torch.unsqueeze(lidar_embeddings[i], 0) @ camera_embeddings.T
        values, predictions = torch.topk(dot_similarity.squeeze(0), max_k)

        queryIdx = i
        queryPos = valid_dataset.dbStruct.utmQ[i]

        predictions = predictions.squeeze()
        matches_at_k = {}
        
        for pred_idx, prediction in enumerate(predictions):
            predictedPos = valid_dataset.dbStruct.utmDb[prediction]
            distance = math.sqrt((queryPos[0] - predictedPos[0])**2 + 
                    (queryPos[1] - predictedPos[1])**2)
            
            for k in recall_Idx:
                if pred_idx < k:
                    if k not in matches_at_k:
                        matches_at_k[k] = distance < CONFIG.threshold_dist
                    else:
                        matches_at_k[k] = matches_at_k[k] or (distance < CONFIG.threshold_dist)
            
        for k in recall_Idx:
            if k in matches_at_k and matches_at_k[k]:
                recall_results[k] += 1

    recall_dict = {}
    
    for k in recall_Idx:
        recall = recall_results[k] / total_queries
        print(f'===: [LSV-Loc EVALUATION] Recall@{k}: {recall*100:.2f}%')
        if rank == 0:
            writer.add_scalar(f"Recall/{k}", recall * 100, epoch)
        recall_dict[f'Recall@{k}'] = recall * 100
        
    return recall_dict
        
# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Training Functions
# --------------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, rank, writer, CONFIG, scaler=None):
    global iteration
    
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader), desc=f"\033[92m===: [LSV-Loc TRAINING] TRAIN: Iteration \033[0m", position=0, leave=True, disable=rank != 0)
    
    for batch in tqdm_object:

        if scaler:
            with autocast():
                batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
                if CONFIG.loss == "BCL":
                    loss = train_step_BCL(model, batch, temperature=CONFIG.temperature)
                else:
                    raise ValueError("===: [LSV-Loc TRAINING] Invalid Loss Function")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            if step == "batch":
                lr_scheduler.step()
                
            count = batch["camera_image"].size(0)
            loss_meter.update(loss.item(), count)

        else:        
            batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
            
            if CONFIG.loss == "BCL":
                loss = train_step_BCL(model, batch, temperature=CONFIG.temperature)
            else:
                raise ValueError("===: [LSV-Loc TRAINING] Invalid Loss Function")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = batch["camera_image"].size(0)
            loss_meter.update(loss.item(), count)
            
        if rank == 0:
            writer.add_scalar("Loss/Train-Batch", loss.item(), iteration)
            iteration += 1
        
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        
    return loss_meter
