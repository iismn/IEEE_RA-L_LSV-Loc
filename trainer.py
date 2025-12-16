import os
import math
import tyro
import wandb
import torch
import random
import datetime
import argparse
import itertools
import importlib
import numpy as np

from os.path import exists, join

# ETC Libraries
from dataclasses import dataclass
from tqdm.autonotebook import tqdm

# Torch Libraries
import torch.distributed as dist
import torch.multiprocessing as mp
import utility.Network.networkTool as networkTool

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Argument Parser
# --------------------------------------------------------------------------------
@dataclass
class Args:
    train_config: str = 'exp_default'
    
args = tyro.cli(Args)

CONFIG = importlib.import_module(f"config.{args.train_config}").CONFIG
get_dataset = importlib.import_module(f"utility.Database.{CONFIG.dataloader}").get_dataset
best_recall = 0.0

def init_wandb(rank):
    if rank == 0:
        wandb.init(
            project="cross-modal-place-recognition",
            mode='online',
            config={
                "image_model_name": "resnet50",
                "epochs": CONFIG.epochs,
            }
        )
    else:

        os.environ["WANDB_MODE"] = "disabled"

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Main Worker
# --------------------------------------------------------------------------------

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def main_worker(rank, world_size, CONFIG):
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] DDP Initialization ---------------------------------------------
    # --------------------------------------------------------------------------------------
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{CONFIG.port_num}',
        timeout=datetime.timedelta(seconds=7200000),
        world_size=world_size,
        rank=rank
    )
    CONFIG.device = torch.device("cuda", rank)
    global best_recall

    if CONFIG.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    if rank == 0:
        print('==================================================================================')
        print('=========== LSV-Loc: LiDAR to StreeetView Cross-Modal Localization ===============')
        print('==================================================================================\n')

        print('===: CONFIG =====================================================================')
        for attr in vars(CONFIG):
            if not attr.startswith('__'):
                value = getattr(CONFIG, attr)
                value_str = str(value)
                print(f"\033[1;33m{attr.center(30)}: {value_str.center(30)}\033[0m")
                
        os.makedirs(CONFIG.expdir, exist_ok=True)
        config_file_path = os.path.join(CONFIG.expdir, 'training_config.txt')
        with open(config_file_path, 'w') as f:
            f.write('===: TRAINING CONFIGURATION ===\n\n')
            for attr in vars(CONFIG):
                if not attr.startswith('__'):
                    value = getattr(CONFIG, attr)
                    f.write(f"{attr}: {str(value)}\n")
            f.write('\n===: END OF CONFIGURATION ===')
    
        print('===: TRAINING ===================================================================')
    
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] Data Loader (DDP) ----------------------------------------------
    # --------------------------------------------------------------------------------------
    train_dataset = get_dataset(mode=CONFIG.train_dataset, CONFIG=CONFIG)
    valid_dataset = get_dataset(mode=CONFIG.val_dataset, CONFIG=CONFIG)
    valid_mini_dataset = get_dataset(mode=CONFIG.test_dataset, CONFIG=CONFIG)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=CONFIG.train_shuffle)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=CONFIG.val_shuffle)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.pin_memory,
        persistent_workers=False,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.pin_memory,
        persistent_workers=True,
        sampler=valid_sampler,
        worker_init_fn=worker_init_fn
    )

    recall_loader = DataLoader(
        dataset=valid_mini_dataset,                           
        batch_size=CONFIG.batch_size//4,  # Using integer division
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.pin_memory,
        persistent_workers=True,
        shuffle=CONFIG.test_shuffle,
        worker_init_fn=worker_init_fn
    )
 
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] Model To Device ------------------------------------------------
    # --------------------------------------------------------------------------------------
    model = importlib.import_module(f"utility.Backbone.{CONFIG.model}").Model(CONFIG)
    
    if CONFIG.resume == True:
        if CONFIG.resume_config == "best":
            model_path = CONFIG.best_resume_model_path
        elif CONFIG.resume_config == "final":
            model_path = CONFIG.final_resume_model_path
        else:
            raise ValueError(f"===: [LSV-Loc] Invalid resume_config: {CONFIG.resume_config}")
        model.load_state_dict(torch.load(model_path, map_location=CONFIG.device))
        print(f"===: [LSV-Loc] Model Loaded from {model_path}")
    
    model.to(CONFIG.device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank], 
        output_device=rank,
        find_unused_parameters=False
    )
    
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] Optimizer & LR Scheduler ---------------------------------------
    # --------------------------------------------------------------------------------------
    encoder_params = list(model.module.encoder.parameters())
    if CONFIG.pooling_fuse:
        params = [
            {"params": encoder_params, "lr": CONFIG.text_encoder_lr},
            {"params": itertools.chain(model.module.projection_fuse.parameters()),
            "lr": CONFIG.head_lr, "weight_decay": CONFIG.weight_decay}
        ]
    else:
        params = [
            {"params": encoder_params, "lr": CONFIG.text_encoder_lr},
            {"params": itertools.chain(model.module.projection_lidar.parameters(),
                                        model.module.projection_camera.parameters()),
            "lr": CONFIG.head_lr, "weight_decay": CONFIG.weight_decay}
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                          patience=CONFIG.patience, factor=CONFIG.factor)
    step = "epoch"
    
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] Wandb / Writer Setting -----------------------------------------
    # --------------------------------------------------------------------------------------
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=CONFIG.logdir)
    
    # --------------------------------------------------------------------------------------
    # [LSV-Loc TRAINING] Training Loop --------------------------------------------------
    # --------------------------------------------------------------------------------------
    best_loss = float('inf')
    for epoch in range(CONFIG.epochs):
        
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        model.train()
        if epoch > 0:
            model.module.update_transforms(current_epoch=epoch, max_epoch=CONFIG.epochs, CONFIG=CONFIG)
        train_loss = networkTool.train_epoch(model, train_loader, optimizer, lr_scheduler, step, rank, writer, CONFIG=CONFIG, scaler=scaler)
        
        model.eval()
        with torch.no_grad():
            model.module.update_transforms_valid(current_epoch=epoch, max_epoch=CONFIG.epochs, CONFIG=CONFIG)
            valid_loss = networkTool.valid_epoch(model, valid_loader, rank, writer, CONFIG=CONFIG)
        
        if rank == 0:
            print(f"\033[38;5;208m===: [LSV-Loc TRAINING] Epoch: {epoch} | Train Loss: {train_loss.avg} | Valid Loss: {valid_loss.avg}\033[0m")
            writer.add_scalar("Loss/Train", train_loss.avg, epoch)
            writer.add_scalar("Loss/Valid", valid_loss.avg, epoch)
            
            torch.save(model.module.state_dict(), CONFIG.final_model_path)
            lr_scheduler.step(valid_loss.avg)
            
            if (epoch + 1) % 1 == 0:
                if rank == 0:
                    print("===: [LSV-Loc EVALUATION] Recall Calculation")
                    model.eval()
                    with torch.no_grad():
                        recall_dict = networkTool.valid_recall(model, valid_mini_dataset, recall_loader, rank, writer, epoch, CONFIG=CONFIG)
                        
                        if 'best_recall' not in globals():
                            best_recall = 0.0
                        if recall_dict["Recall@20"] > best_recall:
                            best_recall = recall_dict["Recall@20"]
                            torch.save(model.module.state_dict(), CONFIG.best_model_path)
                            print(f"===: [LSV-Loc EVALUATION] Improved Recall@20: {best_recall:.4f} - Best Model Saved (Recall)")
                        
                    
        dist.barrier()
    
    if rank == 0:
        writer.close()
        
    dist.destroy_process_group()

def main():
    selected_gpus = CONFIG.CUDA_VISIBLE_DEVICES
    world_size = len(selected_gpus)
    print(f"[DDP] Using GPUs: {selected_gpus}. Spawning {world_size} processes.")
    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, CONFIG),
        join=True
    )

if __name__ == "__main__":
    main()
