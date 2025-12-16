import timm
import math
import numpy as np

import torch
import torch.nn.functional as F
import albumentations as A

import cv2

from torch import nn
from sklearn.neighbors import NearestNeighbors

# Backbone - MatchFormer
from .match_SEA_lite import Matchformer_SEA_lite

# --------------------------------------------------------------------------------
# [SVR-LipLoc TRAINING] Input Transforms
# --------------------------------------------------------------------------------

def convert_uniform_to_nonuniform_lidar(image, p=2.0, **kwargs):

    if image.ndim == 3:
        image = image[:, :, 0]
        
    h, w = image.shape
    valid = image[(h//2 - 64):(h//2 + 64), :]  # shape (128, w)
    
    x = np.linspace(0, 1, 128)
    new_x = 1 - (1 - x)**p
    target_angles = -25 + new_x * 40
    sample_rows = (target_angles + 25) / 40 * (128 - 1)
    
    new_valid = np.zeros((128, w), dtype=valid.dtype)
    for col in range(w):
        col_data = valid[:, col]
        new_valid[:, col] = np.interp(sample_rows, np.arange(128), col_data)
    
    out_single = np.zeros((256, w), dtype=new_valid.dtype)
    start_row = (256 - 128) // 2  # 64
    out_single[start_row:start_row+128, :] = new_valid

    out_image = np.stack([out_single, out_single, out_single], axis=-1)
    
    return out_image

def random_horizontal_roll(image, **kwargs):
    shift = np.random.randint(0, image.shape[1])
    return np.roll(image, shift=shift, axis=1)

def simulate_lidar_fov_channel(
        img: np.ndarray,
        out_channels: int,
        in_channels: int = 128,
        in_fov_deg: float = 45.0,
):
    """
    img            : (H=256, W=1024[, C]) – 원본 LiDAR intensity 이미지
    out_channels   : 시뮬레이션할 수직 채널 개수 (예: 64, 32 …)
    out_fov_deg    : 시뮬레이션할 수직 FoV 각도 (예: 22.5)
    in_channels    : 원본 유효 채널 (=128)
    in_fov_deg     : 원본 FoV (=45.0°)
    crop_mode      : 어떤 부분을 잘라낼지 ("center"=대칭, "top"=위쪽, "bottom"=아래쪽)
    ------------------------------------------------------------
    반환값         : (256, 1024[, C]) –  패딩 포함 복구된 이미지
    """
    H0, W0 = img.shape[:2]
    img = img[:, :, 0]
    img = cv2.resize(img, (1024, 256), interpolation=cv2.INTER_LINEAR)
    assert img.shape[0] == 256 and img.shape[1] == 1024, "Expect 256×1024 input"

    # 1) 유효 영역(128×1024)만 추출
    pad = (img.shape[0] - in_channels) // 2        # =64
    valid = img[pad:pad + in_channels]             # shape (128, 1024[, C])
    valid_flip = valid[::-1, :]
    # 2) 원하는 FOV 구간 자르기
    # ──────────────── 예시 선언 ────────────────
    # 원본: 128채널, FoV –22.5°…+22.5°
    in_channels   = 128
    in_fov_deg    = 22.5
    # 시뮬레이션하고 싶은 구간
    out_fov_start = -12.25  # 시작 각도
    out_fov_end   = +12.25   # 끝 각도
    # ─────────────────────────────────────────
    
    deg_per_ch = in_fov_deg / in_channels   # 채널 당 각도
    # in FoV의 중간(–in_fov/2)에서 각 채널이 매핑되는 절대 각도
    # 인덱스 0 → angle = -in_fov/2, 인덱스 in_channels-1 → +in_fov/2
    mid_offset = - in_fov_deg / 2.0

    # 시작 각도/끝 각도를 채널 인덱스로 변환
    start_ch = int(round((out_fov_start - mid_offset) / deg_per_ch))
    end_ch   = int(round((out_fov_end   - mid_offset) / deg_per_ch)) + 1

    # 경계 clamp
    start_ch = np.clip(start_ch, 0, in_channels-1)
    end_ch   = np.clip(end_ch,   1, in_channels)

    # valid[ start_ch : end_ch ] 을 crop
    # cropped = valid[start_ch:end_ch]
    cropped = valid_flip[start_ch:end_ch, :]
    cropped = cropped[::-1, :]
    # 3) 세로 리샘플링 → out_channels
    # cv2 입력은 (H, W[, C]), INTER_LINEAR이면 float-intensity에 무해
    resized = cv2.resize(cropped, (cropped.shape[1], out_channels),
                         interpolation=cv2.INTER_LINEAR)

    # 4) 네트워크가 기대하는 128 채널로 다시 업샘플
    restored = cv2.resize(resized, (resized.shape[1], in_channels),
                          interpolation=cv2.INTER_LINEAR)

    # 5) 256 × 1024로 zero-padding
    out_img = np.zeros_like(img)
    out_img[pad:pad + in_channels] = restored
    out_img = np.stack([out_img, out_img, out_img], axis=-1)
    out_img = cv2.resize(out_img, (W0, H0), interpolation=cv2.INTER_LINEAR)
    
    # out_img 저장 (디버깅 또는 시각화용)
    save_path = "/tmp/simulated_lidar_out_img.png"
    # cv2.imwrite(save_path, out_img)
    # print(f"Simulated LiDAR image saved to {save_path}")
    
    return out_img

def input_transforms_ViT_LiDAR(mode, size, config = None):
    if mode == "train":
        return A.Compose([
            A.Lambda(name="RandomHorizontalRoll", image=random_horizontal_roll, p=0.5),
            A.Normalize(max_pixel_value=255.0),
        ])
    elif mode == "LIDAR_simulation":
        return A.Compose([
            A.Lambda(
                name="SimulateLidar",
                image=lambda img, **kw: simulate_lidar_fov_channel(
                    img,
                    in_channels=128, in_fov_deg=45,
                    out_channels=config.LIDAR_Channel
                ),
                p=1.0
            ),
            A.Normalize(max_pixel_value=255.0),
        ])
    else:
        return A.Compose([
            A.Normalize(max_pixel_value=255.0),
        ])

def input_transforms_ViT_SVR(mode, size, config = None):
    if mode == "train":
        return A.Compose(
            [   
                A.Lambda(
                    name="RandomHorizontalRoll",
                    image=random_horizontal_roll,
                    p=0.5
                ),
                A.Normalize(max_pixel_value=255.0),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(max_pixel_value=255.0),
            ]
        )

def input_transforms_dynamic_LiDAR(mode, size, current_epoch, max_epoch=50):

    def lin_schedule(init, final, epoch, max_epoch):
        return init + (final - init) * (epoch / max_epoch)
    
    brightness_contrast_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    gamma_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    noise_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    
    print(f'===: [SVR-LipLoc TRAINING] Dynamic Augmentation Probabilities : {brightness_contrast_prob:.2f}, {gamma_prob:.2f}, {noise_prob:.2f}')
    
    if mode == "train":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=brightness_contrast_prob),
            A.RandomGamma(gamma_limit=(80, 120), p=gamma_prob),
            A.GaussNoise(var_limit=(10.0, 50.0), p=noise_prob),
            A.Lambda(
                name="UniformToNonUniformLiDAR",
                image=lambda img, **kwargs: convert_uniform_to_nonuniform_lidar(img, p=2.0, **kwargs),
                p=noise_prob
            ),
            A.Normalize(max_pixel_value=255.0),
        ])
    else:
        return A.Compose([
            A.Normalize(max_pixel_value=255.0),
        ])

def input_transforms_dynamic_SVR(mode, size, current_epoch, max_epoch=50):

    def lin_schedule(init, final, epoch, max_epoch):
        return init + (final - init) * (epoch / max_epoch)
    
    brightness_contrast_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    gamma_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    noise_prob = lin_schedule(0.1, 0.8, current_epoch, max_epoch)
    
    if mode == "train":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=brightness_contrast_prob),
            A.RandomGamma(gamma_limit=(80, 120), p=gamma_prob),
            A.GaussNoise(var_limit=(10.0, 50.0), p=noise_prob),
            A.Normalize(max_pixel_value=255.0),
        ])
    else:
        return A.Compose([
            A.Normalize(max_pixel_value=255.0),
        ])

# --------------------------------------------------------------------------------
# [SVR-LipLoc TRAINING] Sinkhorn Algorithm
# --------------------------------------------------------------------------------

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

# --------------------------------------------------------------------------------
# [SVR-LipLoc TRAINING] Image Encoders
# --------------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable, embed_dim=768, project_dim=2048, transforms_dict=None, model_hub="timm"):
        super().__init__()

        self.model_hub = model_hub
        
        if self.model_hub == "timm":
            self.encoder = timm.create_model(model_name.lower(), pretrained, num_classes=0, global_pool="avg")
        elif self.model_hub == "torch":
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
            if hasattr(self.encoder, "global_pool"):
                self.encoder.global_pool = nn.Identity()
        elif self.model_hub == "custom":
            if model_name == 'match_SEA_lite':
                self.encoder = Matchformer_SEA_lite(img_size=518)
                
                # load pretrained weights for Matchformer_SEA_lite
                ckpt_path = "/home/iris/Workspace/Sangmin/SVR_Loc/utility/Backbone/pretrained_net/outdoor-lite-SEA.ckpt"
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                self.encoder.load_state_dict(state_dict, strict=False)
                
                print("=====: [SVR-LipLoc INFO] Using Matchformer SEA Lite ===")
            else:
                raise ValueError(f"Unknown model_hub: {self.model_hub}")

        self.fc = nn.Linear(embed_dim, project_dim)
        self.transforms_dict = transforms_dict if transforms_dict is not None else {}
        # self.patch_size = self.encoder.patch_embed.proj.kernel_size[0]
        self.patch_size = 14
        
        if not trainable:
            print("===: [SVR-LipLoc INFO] Freezing Image Encoder ===")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            print("===: [SVR-LipLoc INFO] Training Image Encoder ===")
            for p in self.encoder.parameters():
                p.requires_grad = True

    def forward(self, input_tensor, modality="camera"):

        if modality in self.transforms_dict and self.transforms_dict[modality] is not None:
            transform = self.transforms_dict[modality]
            processed_images = []
            for img in input_tensor:

                img_np = img.permute(1, 2, 0).cpu().numpy()
                
                if self.model_hub == "torch":
                    H, W = img_np.shape[0], img_np.shape[1]
                    new_H = (H // self.patch_size) * self.patch_size
                    new_W = (W // self.patch_size) * self.patch_size
                    if (H != new_H) or (W != new_W):
                        img_np = np.array(
                            A.Resize(new_H, new_W)(image=img_np)['image']
                        )
                        
                img_np = transform(image=img_np)['image']
                img_tensor = torch.tensor(img_np).permute(2, 0, 1).float()

                processed_images.append(img_tensor)

            x = torch.stack(processed_images, dim=0).to(next(self.encoder.parameters()).device)
        
        y1 = self.encoder(x)
        y2 = self.fc(y1)
        return y2

# --------------------------------------------------------------------------------------
# [SVR-LipLoc Pooling] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

class ProjectionBlock(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        residual = x
        x = self.gelu(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, final_embedding_dim, projection_dim, dropout, num_projection_layer=1):
        super().__init__()
        self.projection = nn.Linear(projection_dim, final_embedding_dim)
        self.layers = nn.ModuleList([
            ProjectionBlock(final_embedding_dim, dropout) for _ in range(num_projection_layer)
        ])
    
    def forward(self, x):
        x = self.projection(x)
        for block in self.layers:
            x = block(x)
        return x

# --------------------------------------------------------------------------------------
# [SVR-LipLoc Model] -------------------------------------------------------------------
# --------------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        
        self.device=CONFIG.device
        self.temperature=CONFIG.temperature
        self.pooling_fuse = CONFIG.pooling_fuse
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.camera_transform = input_transforms_ViT_SVR(mode=CONFIG.mode, size=CONFIG.image_size, config=CONFIG)
        self.lidar_transform = input_transforms_ViT_LiDAR(mode=CONFIG.mode, size=CONFIG.image_size, config=CONFIG)
        self.transforms_dict = {"camera": self.camera_transform,"lidar": self.lidar_transform}
        
        # -----------------------------------------
        # [SVR-LipLoc TRAINING] Encoder Initialization
        # -----------------------------------------
        self.encoder = ImageEncoder(model_name=CONFIG.model_name, pretrained=CONFIG.pretrained, trainable=CONFIG.trainable, 
                                            embed_dim=CONFIG.embedding_dim, project_dim=CONFIG.projection_dim, 
                                            model_hub=CONFIG.model_hub, transforms_dict=self.transforms_dict)
                    
        # -----------------------------------------
        # [SVR-LipLoc TRAINING] PRJ Head Initialization
        # -----------------------------------------
        if self.pooling_fuse == True:
            self.projection_fuse = ProjectionHead(projection_dim=CONFIG.projection_dim, final_embedding_dim=CONFIG.final_embedding_dim, dropout=CONFIG.dropout, num_projection_layer=CONFIG.num_projection_layer)
        elif self.pooling_fuse == False:
            self.projection_camera = ProjectionHead(projection_dim=CONFIG.camera_projection_dim, final_embedding_dim=CONFIG.final_embedding_dim, dropout=CONFIG.dropout, num_projection_layer=CONFIG.num_projection_layer)
            self.projection_lidar = ProjectionHead(projection_dim=CONFIG.lidar_projection_dim, final_embedding_dim=CONFIG.final_embedding_dim, dropout=CONFIG.dropout, num_projection_layer=CONFIG.num_projection_layer)
 

    def update_transforms(self, current_epoch, max_epoch=50, CONFIG=None):

        self.camera_transform = input_transforms_dynamic_SVR(mode="train", size=CONFIG.image_size, current_epoch=current_epoch, max_epoch=max_epoch)
        self.lidar_transform  = input_transforms_dynamic_LiDAR(mode="train", size=CONFIG.image_size, current_epoch=current_epoch, max_epoch=max_epoch)
        self.transforms_dict = {"camera": self.camera_transform, "lidar": self.lidar_transform}

    def update_transforms_valid(self, current_epoch, max_epoch=50, CONFIG=None):

        self.camera_transform = input_transforms_dynamic_SVR(mode="val", size=CONFIG.image_size, current_epoch=current_epoch, max_epoch=max_epoch)
        self.lidar_transform  = input_transforms_dynamic_LiDAR(mode="val", size=CONFIG.image_size, current_epoch=current_epoch, max_epoch=max_epoch)
        self.transforms_dict = {"camera": self.camera_transform, "lidar": self.lidar_transform}

    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        camera_image_features = self.encoder(batch["camera_image"], modality="camera")
        lidar_image_features = self.encoder(batch["lidar_image"], modality="lidar")
        
        if self.pooling_fuse == True:
            camera_image_embeddings = self.projection_fuse(camera_image_features)
            lidar_image_embeddings = self.projection_fuse(lidar_image_features)
        elif self.pooling_fuse == False:
            camera_image_embeddings = self.projection_camera(camera_image_features)
            lidar_image_embeddings = self.projection_lidar(lidar_image_features)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        image_features1 = F.normalize(lidar_image_embeddings, dim=-1)
        image_features2 = F.normalize(camera_image_embeddings, dim=-1)

        logits_per_image1 = self.logit_scale * torch.matmul(image_features1, image_features2.T)
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(image_features1.shape[0], dtype=torch.long, device=image_features1.device)
        loss = (loss_fn(logits_per_image1, labels) + loss_fn(logits_per_image2, labels)) / 2
        
        return loss.mean()

    def camera_embeddings(self, batch):
        image_features = self.encoder(batch["camera_image"].to(self.device), modality="camera")
        if self.pooling_fuse == True:
            image_embeddings = self.projection_fuse(image_features)
        else:
            image_embeddings = self.projection_camera(image_features)
        return image_embeddings

    def lidar_embeddings(self, batch):
        image_features = self.encoder(batch["lidar_image"].to(self.device), modality="lidar")

        if self.pooling_fuse == True:
            image_embeddings = self.projection_fuse(image_features)
        else:
            image_embeddings = self.projection_lidar(image_features)
        return image_embeddings

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_topk(query_image_embeddings, lidar_image_embeddings, n=1):
    dot_similarity = query_image_embeddings @ lidar_image_embeddings.T
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    return values, indices