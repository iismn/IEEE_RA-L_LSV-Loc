import torch
from pathlib import Path
from datetime import datetime

class CONFIG:
    
    # GENERAL CONFIG
    mode = "LIDAR_simulation"                     # train / val / test
    debug = False
    expid = Path(__file__).stem
    current_date = '04160633'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # COMPUTING CONFIG 
    batch_size = 10                             # GPU Batch Size
    num_workers = 24                            # CPU Loader Workers
    image_size = 518                            # Input image size      
    mixed_precision = True                      # Automatic Mixed Precision (AMP) Training
    
    # DATA DISTRIBUTED PARALLEL (DDP) CONFIG
    port_num = 12346
    pin_memory = True
    CUDA_VISIBLE_DEVICES = [0]
    
    # TRAINING CONFIG
    epochs = 50
    factor = 0.8
    patience = 1
    dropout = 0.3
    head_lr = 1e-4
    max_length = 200
    temperature = 1.0
    weight_decay = 1e-3
    text_encoder_lr = 1e-5
    image_encoder_lr = 1e-5
    final_embedding_dim = 768
    
    # MODEL CONFIG 
    model = "CLIP_AIO"
    model_hub = 'torch'                         # 'torch' / 'huggingface' / 'timm'
    model_name = 'dinov2_vitb14_reg'            # 'dinov2_vitb14_reg' / 'vit_base_patch14_reg4_dinov2.lvd142m'
    resume_model = 'best'                      # 'best' / 'final'

    trainable = True
    embedding_dim = 768                         # Encoder Embedding Dimension
    projection_dim = 8192
    
    trainable_blocks = 0
    threshold_dist = 25                         # Positive Distance Threshold
    pretrained = True                           # Pretrained Weights Enable
    num_projection_layer = 1                    # Projection Layer Block

    # POOLING CONFIG
    pooling = "Projection"
    pooling_fuse = True

    lidar_embedding_dim = 768                   # Poosing Fuse = False
    camera_embedding_dim = 768                  # Poosing Fuse = False
    lidar_projection_dim = 8192                 # Poosing Fuse = False
    camera_projection_dim = 8192                # Poosing Fuse = False

    # LOSS CONFIG
    loss = "BCL"                    # BCL - Batch InfoNCE Loss
    temperature = 0.1
    
    # DATASET CONFIG
    dataloader = "strvDataset_AIO"
    data_path = "../dataset/SVR_Dataset_Sync/" #TODO: Set path
    
    # LOGGING CONFIG
    expdir = f"result/{current_date}_{model_name}/"
    logdir = f"result/{current_date}_{model_name}/log/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    
    # RESUME CONFIG
    resume = False
    resume_date = ""
    resume_config = "best"
    expdir_resume = f"result/{resume_date}_{model_name}/"
    best_resume_model_path = f"{expdir_resume}best.pth"
    final_resume_model_path = f"{expdir_resume}model.pth"
    
    # DATASET CONFIG
    train_dataset = 'train'
    train_shuffle = True
    
    val_dataset = 'val'
    val_shuffle = False
    
    test_dataset = 'val'
    test_seq = 'Roundabout01'                     # ComplexUrban05 / ComplexUrban08 / Dunsan / Roundabout01 / All
    test_shuffle = False
    
    LIDAR_name = 'OS2-32'                # LIDAR Sensor Name
    LIDAR_simulation = False
    LIDAR_Channel = 32