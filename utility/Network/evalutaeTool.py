import gc
import torch
import tifffile
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R
import albumentations as A

class EvalTool:
    def __init__(self, model=None, device=None, patch_size=14):
        """
        model: SVRMatcher_Test.model
        device: torch.device
        patch_size: int
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size

    @staticmethod
    def process_path(path: str) -> str:
        if path.startswith('./'):
            path = path[2:]
        return f"dataset/SVR_Dataset_Sync/{path}"

    def attention_map(self, image_tensor: torch.Tensor, layer_idx: int=None) -> np.ndarray:
        """
        Returns a (h_patches, w_patches) numpy attention map.
        """
        # BCHW support
        img = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        vit = self.model.encoder.encoder
        p = vit.patch_embed.proj.kernel_size[0]
        B, C, H, W = img.shape
        new_H, new_W = (H//p)*p, (W//p)*p
        h_patches, w_patches = new_H//p, new_W//p
        layer_idx = layer_idx or len(vit.blocks) - 1
        Rtok = getattr(vit, "register_tokens", torch.empty(1,0)).shape[1]

        attn_maps = []
        def hook_fn(_, inp, out):
            attn_maps.append(out.detach().cpu())
        handle = vit.blocks[layer_idx].attn.attn_drop.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = vit(img[..., :new_H, :new_W])
        handle.remove()
        if not attn_maps:
            raise RuntimeError("No attention maps captured.")

        attn_all = attn_maps[0][0]      # (heads, tokens, tokens)
        cls_to_p = attn_all.sum(0)[0, 1 + Rtok:]  # Sum over heads, CLS->patch attention
        
        # Debug print
        # print(f"Attention map stats: shape={cls_to_p.shape}, min={cls_to_p.min():.4f}, max={cls_to_p.max():.4f}, mean={cls_to_p.mean():.4f}")
        
        attention_map = cls_to_p.reshape(h_patches, w_patches).numpy()
        return attention_map

    def preprocess_img_tensor(self, img, patch_size=14):
        """
        img: torch.Tensor or numpy.ndarray, shape (C, H, W) or (H, W, C)
        patch_size: int, patch size for ViT
        device: torch.device or None
        Returns: img_tensor (1, C, new_H, new_W) on device
        """
        import albumentations as A
        
        transform = A.Compose([
            A.Normalize(max_pixel_value=255.0),
        ])
        
        # img가 numpy라면 그대로, tensor라면 numpy로 변환
        if isinstance(img, np.ndarray):
            img_np = img  # HxWxC
        else:
            img_np = img.permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC

        H, W = img_np.shape[0], img_np.shape[1]
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size
        if (H != new_H) or (W != new_W):
            img_np = np.array(
                A.Resize(new_H, new_W)(image=img_np)['image']
            )
            
        img_np = transform(image=img_np)['image']
            
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).float().unsqueeze(0)
        if self.device is not None:
            img_tensor = img_tensor.to(self.device)
            
        grid_size = (new_H // patch_size, new_W // patch_size)
        
        return img_tensor, grid_size

    @staticmethod
    def plot_attention_overlay(img_tensor: torch.Tensor,
                               attn_map_resized: np.ndarray,
                               alpha: float=0.5, cmap: str='jet'):
        if img_tensor.dim() == 4:
            img = img_tensor.squeeze(0).cpu().numpy()
        else:
            img = img_tensor.cpu().numpy()
        img = np.transpose(img, (1,2,0))
        mi, ma = img.min(), img.max()
        if ma > 1.0:
            img = (img - mi)/(ma - mi + 1e-8)
        cmap_rgb = plt.get_cmap(cmap)(attn_map_resized)[...,:3]
        overlay = np.clip((1-alpha)*img + alpha*cmap_rgb, 0,1)
        plt.figure(figsize=(12,6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    @staticmethod
    def mutual_nearest_neighbors(database_tokens: np.ndarray,
                                 query_tokens: np.ndarray,
                                 grid_size: tuple):

        knn_q2d = NearestNeighbors(n_neighbors=1)
        knn_q2d.fit(database_tokens)
        dist_q2d, idx_q2d = knn_q2d.kneighbors(query_tokens)
        idx_q2d = idx_q2d.squeeze(1)  # shape: (num_query_patches,)

        # 2. db→query (query_tokens에서 database_tokens의 최근접 이웃)
        knn_d2q = NearestNeighbors(n_neighbors=1)
        knn_d2q.fit(query_tokens)
        dist_d2q, idx_d2q = knn_d2q.kneighbors(database_tokens)
        idx_d2q = idx_d2q.squeeze(1)  # shape: (num_db_patches,)

        # 3. Mutual Nearest Neighbor 매칭
        mutual_matches = []
        for db_idx, query_idx in enumerate(idx_d2q):
            if idx_q2d[query_idx] == db_idx:
                mutual_matches.append((query_idx, db_idx))

        # 필요하다면 매칭 결과를 numpy array로 변환
        mutual_matches = np.array(mutual_matches)

        for query_idx, db_idx in mutual_matches:
            rowA, colA = EvalTool.idx_to_source_position(query_idx, grid_size, 14)
            rowB, colB = EvalTool.idx_to_source_position(db_idx, grid_size, 14)

            rowA = float(rowA)
            colA = float(colA)
            rowB = float(rowB)
            colB = float(colB)

            xyA = (colA, rowA)  # query
            xyB = (colB, rowB)  # database
                
        return mutual_matches, xyA, xyB


    @staticmethod
    def idx_to_source_position(idx: int, grid_size: tuple, patch_size: float=14):
        row = (idx//grid_size[1])*patch_size + patch_size/2
        col = (idx%grid_size[1])*patch_size + patch_size/2
        return float(row), float(col)

    @staticmethod
    def single_point_from_range_image(u: int, v: int, depth: float, range_image: np.ndarray):
        numCols, numRows = range_image.shape[1], range_image.shape[0]
        az = -np.pi + u*(2*np.pi/(numCols-1))
        el = np.deg2rad(22.5) - v*(np.deg2rad(45)/(numRows-1))
        x = depth*np.cos(el)*np.sin(az)
        y = depth*np.cos(el)*np.cos(az)
        z = depth*np.sin(el)
        return np.array([x,y,z])

    @staticmethod
    def rotation_y(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    @staticmethod
    def project_equirectangular(P_cam: np.ndarray, width: int, height: int):
        x, y, z = P_cam[0], P_cam[2], P_cam[1]
        r = np.linalg.norm(P_cam)
        if r < 1e-8:
            return np.array([width/2, height/2])
        theta = np.arctan2(x, z)
        phi = np.arcsin(y/r)
        u = (theta + np.pi)/(2*np.pi)*width
        v = (np.pi/4 - phi)/(np.pi/2)*height
        return np.array([u, v])

    @staticmethod
    def residuals(params, world_points, image_points, width, height, cam_height):
        tx, ty, yaw = params
        R_h = EvalTool.rotation_y(yaw)
        t = np.array([tx, cam_height, ty])
        res = []
        for P_w, uv in zip(world_points, image_points):
            p = P_w - t
            u_proj, v_proj = EvalTool.project_equirectangular(R_h @ p, width, height)
            res += [u_proj - uv[0], v_proj - uv[1]]
        res = np.array(res)
        if res.size:
            errs = res.reshape(-1,2)
            # print(f"Mean residual: {np.mean(np.linalg.norm(errs,axis=1)):.4f}")
        return res

    @staticmethod
    def collect_world_and_image_points(filtered_matches, grid_size, range_image,
                                       scale_factor=1.5, plot=True, save_path=None):
        world_pts, img_pts, us, vs = [], [], [], []
        for q, d in filtered_matches:
            rB, cB = EvalTool.idx_to_source_position(d, grid_size)
            img_pts.append([cB, rB])
            rA, cA = EvalTool.idx_to_source_position(q, grid_size)
            u, v = int(round(cA)), int(round(rA - 64*scale_factor))
            if 0 <= v < range_image.shape[0] and 0 <= u < range_image.shape[1]:
                depth = range_image[v, u]
                world_pts.append(EvalTool.single_point_from_range_image(u, v, depth, range_image))
                us.append(u); vs.append(v)
        
        if plot or save_path:
            plt.figure(figsize=(12,6))
            plt.imshow(range_image, cmap='viridis')
            plt.scatter(us, vs, c='r', s=60, label='Matched Points')
            plt.colorbar(label='Depth (m)')
            plt.legend()
            plt.title('Range Image with Feature Matches')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close()
            elif plot:
                plt.show()
            else:
                plt.close()
                
        return world_pts, img_pts
