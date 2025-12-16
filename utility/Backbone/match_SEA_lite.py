import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class RMacPool2d(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        """
        L: number of scales (levels). 추천 값 3.
        """
        super().__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C)
        """
        B, C, H, W = x.shape
        R = []  # list of region‐pooled tensors

        # min side
        min_side = min(H, W)

        for l in range(1, self.L + 1):
            # region size for level l
            region_size = math.floor(2 * min_side / (l + 1))
            # stride between region centers
            if l + 1 > 1:
                step_h = (H - region_size) / l
                step_w = (W - region_size) / l
            else:
                step_h = step_w = 0

            for i in range(l + 1):
                for j in range(l + 1):
                    # top-left corner
                    start_h = int(round(i * step_h))
                    start_w = int(round(j * step_w))
                    end_h = min(start_h + region_size, H)
                    end_w = min(start_w + region_size, W)

                    # slice region and max-pool
                    region = x[:, :, start_h:end_h, start_w:end_w]  # (B, C, rh, rw)
                    # if region too small, pad or skip
                    if region.numel() == 0:
                        continue
                    # adaptive_max_pool → (B, C, 1, 1)
                    pooled = F.adaptive_max_pool2d(region, (1, 1))
                    # flatten to (B, C)
                    vec = pooled.view(B, C)
                    # L2 normalize per sample
                    vec = vec / (vec.norm(dim=1, keepdim=True) + self.eps)
                    R.append(vec)

        if not R:
            # fallback to global max
            out = F.adaptive_max_pool2d(x, 1).view(B, C)
            return out / (out.norm(dim=1, keepdim=True) + self.eps)

        # stack → (num_regions, B, C) → sum over regions
        R_stack = torch.stack(R, dim=0)    # (R, B, C)
        R_sum = R_stack.sum(dim=0)         # (B, C)
        # final L2 normalize
        R_out = R_sum / (R_sum.norm(dim=1, keepdim=True) + self.eps)
        return R_out

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,H,W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, cross= False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.cross = cross

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.cross == True:
            MiniB = B // 2
            #cross attention
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q1,q2 = q.split(MiniB)

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
            k1, k2 = kv[0].split(MiniB)
            v1, v2 = kv[1].split(MiniB)

            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            x1 = (attn1 @ v2).transpose(1, 2).reshape(MiniB, N, C)
            x2 = (attn2 @ v1).transpose(1, 2).reshape(MiniB, N, C)

            x = torch.cat([x1, x2], dim=0)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, cross = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, cross= cross)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Positional(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, with_pos=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = Positional(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        _, _, H, W = x.shape       
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W

class AttentionBlock(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dims=128, patch_size=7, num_heads=1, mlp_ratios=4, sr_ratios=8,
                 qkv_bias=True, drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),stride=2, depths=1, cross=[False,False,True]):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride = stride, in_chans=in_chans,
                                              embed_dim=embed_dims)
        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,sr_ratio= sr_ratios,
            drop=drop_rate, drop_path=0, norm_layer=norm_layer, cross=cross[i])
            for i in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W  = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x,H,W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        """
        p: 초기 지수값. 
        eps: numerical stability.
        learn_p: True 이면 p 를 학습 가능하게 만듦.
        """
        super().__init__()
        self.eps = eps
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)

class FPNGeMProjector(nn.Module):
    def __init__(self, in_channels_list, out_channels=512, gem_p=3.0, learn_p=True):
        """
        in_channels_list: [c4_ch, c3_ch, c2_ch, c1_ch]
        out_channels:      FPN 후 최종 채널 수 (임베딩 차원)
        gem_p:             GeM 초기 p 값
        learn_p:          GeM p 를 학습할지 여부
        """
        super().__init__()
        c4, c3, c2, c1 = in_channels_list

        # lateral 1×1 conv 로 모두 out_channels 로 맞춤
        self.lateral4 = conv1x1(c4, out_channels)
        self.lateral3 = conv1x1(c3, out_channels)
        self.lateral2 = conv1x1(c2, out_channels)
        self.lateral1 = conv1x1(c1, out_channels)

        # smoothing 3×3 conv (optional)
        self.smooth3 = conv3x3(out_channels, out_channels)
        self.smooth2 = conv3x3(out_channels, out_channels)
        self.smooth1 = conv3x3(out_channels, out_channels)

        # GeM pool instead of GAP
        self.pool = GeM(p=gem_p, learn_p=learn_p)

    def up_add(self, x, y):
        # x: 높은 레벨 feature; y: lateral feature
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, c4, c3, c2, c1):
        # 1) top pyramid
        p4 = self.lateral4(c4)                   # (B, out_c, H4, W4)

        # 2) top-down + lateral
        p3 = self.up_add(p4, self.lateral3(c3))
        p3 = self.smooth3(p3)

        p2 = self.up_add(p3, self.lateral2(c2))
        p2 = self.smooth2(p2)

        p1 = self.up_add(p2, self.lateral1(c1))
        p1 = self.smooth1(p1)                    # (B, out_c, H1, W1)

        # 3) GeM pooling → (B, out_c, 1, 1) → reshape → (B, out_c)
        out = self.pool(p1).view(p1.size(0), -1)
        return out

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

class Matchformer_SEA_lite(nn.Module):
    def __init__(self, img_size=518, in_chans=3, embed_dims=[128, 192, 256, 512], num_heads=[1, 2, 4, 8],sr_ratios=[8,4,2,1]
                ,stage1_cross = [False,False,False],stage2_cross = [False,False,False],stage3_cross = [False,False,False],stage4_cross = [False,False,False]):
        super().__init__()
        #Attention
        self.AttentionBlock1 =  AttentionBlock(img_size=img_size // 2, patch_size=7, num_heads= num_heads[0], mlp_ratios=4,  in_chans=in_chans,
                                              embed_dims=embed_dims[0],stride=4,sr_ratios=sr_ratios[0],depths=3, cross=stage1_cross)
        self.AttentionBlock2 =  AttentionBlock(img_size=img_size // 4, patch_size=3, num_heads= num_heads[1], mlp_ratios=4,  in_chans=embed_dims[0],
                                              embed_dims=embed_dims[1],stride=2,sr_ratios=sr_ratios[1],depths=3, cross=stage2_cross)
        self.AttentionBlock3 =  AttentionBlock(img_size=img_size // 16,patch_size=3, num_heads= num_heads[2], mlp_ratios=4,  in_chans=embed_dims[1],
                                              embed_dims=embed_dims[2],stride=2,sr_ratios=sr_ratios[2],depths=3, cross=stage3_cross)
        self.AttentionBlock4 =  AttentionBlock(img_size=img_size // 32,patch_size=3, num_heads= num_heads[3], mlp_ratios=4,  in_chans=embed_dims[2],
                                              embed_dims=embed_dims[3],stride=2,sr_ratios=sr_ratios[3],depths=3, cross=stage4_cross)
                                            
        #FPN
        self.layer4_outconv = conv1x1(embed_dims[3], embed_dims[3])
        self.layer3_outconv = conv1x1(embed_dims[2], embed_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(embed_dims[3], embed_dims[3]),
            nn.BatchNorm2d(embed_dims[3]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[3], embed_dims[2]),
        )
        self.layer2_outconv = conv1x1(embed_dims[1], embed_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(embed_dims[2], embed_dims[2]),
            nn.BatchNorm2d(embed_dims[2]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[2], embed_dims[1]),
        )
        self.layer1_outconv = conv1x1(embed_dims[0], embed_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(embed_dims[1], embed_dims[1]),
            nn.BatchNorm2d(embed_dims[1]),
            nn.LeakyReLU(),
            conv3x3(embed_dims[1], embed_dims[0]),
        )

        self.fpn_gem = FPNGeMProjector(
            in_channels_list=[512,256,192,128],
            out_channels=512,
            gem_p=3.0,
            learn_p=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # stage 1 # 1/4        
        x = self.AttentionBlock1(x)
        out1 = x
        # stage 2 # 1/8
        x = self.AttentionBlock2(x)     
        out2 = x                    
        # stage 3 # 1/16
        x = self.AttentionBlock3(x)                          
        out3 = x
        # stage 3 # 1/32
        x = self.AttentionBlock4(x)                          
        out4 = x 
        
        #FPN
        c4_out = self.layer4_outconv(out4)
        _,_,H,W = out3.shape
        c4_out_2x = F.interpolate(c4_out, size =(H,W), mode='bilinear', align_corners=True)
        c3_out = self.layer3_outconv(out3)
        _,_,H,W = out2.shape
        c3_out = self.layer3_outconv2(c3_out +c4_out_2x)
        c3_out_2x = F.interpolate(c3_out, size =(H,W), mode='bilinear', align_corners=True)
        c2_out = self.layer2_outconv(out2)
        _,_,H,W = out1.shape
        c2_out = self.layer2_outconv2(c2_out +c3_out_2x)
        c2_out_2x = F.interpolate(c2_out, size =(H,W), mode='bilinear', align_corners=True)
        c1_out = self.layer1_outconv(out1)
        c1_out = self.layer1_outconv2(c1_out+c2_out_2x)
        
        print("c4_out.shape",c4_out.shape)
        print("c3_out.shape",c3_out.shape)
        print("c2_out.shape",c2_out.shape)
        print("c1_out.shape",c1_out.shape)
        
        # flatten spatial dims and move channels to last dim
        B, C, H, W = c1_out.shape
        x = c1_out.view(B, C, -1).permute(0, 2, 1)   # [B, N, C] where N = H*W

        # apply a ProjectionBlock to each token
        proj = ProjectionBlock(embedding_dim=C, dropout=0.1).to(x.device)
        x = proj(x)                                 # [B, N, C]

        # pool over the N tokens to get a single C‐dim vector per example
        x = x.mean(dim=1)                           # [B, C]

        # finally map to 512‐dim
        fc_out = nn.Linear(C, 512).to(x.device)
        global_emb = fc_out(x)                      # [B, 512]
        
        return global_emb