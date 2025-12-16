import os
import gc
import cv2
import sys
import math
import tyro
import torch
import tifffile
import importlib
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from dataclasses import dataclass
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from tqdm.autonotebook import tqdm
from scipy.optimize import least_squares

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from config.strv_eval import CONFIG as args

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append('.')

import utility.Network.evalutaeTool as evalTool

# Load configuration from config/exp_strv_Eval.py instead of using tyro
try:
    print(f"Configuration loaded from config/strv_config_PnP.py")
except ImportError:
    # Fallback to tyro if the import fails
    @dataclass
    class Args:
        expid: str = 'exp_strv_Eval'
    args = tyro.cli(Args)
    print(f"Using command line arguments with expid: {args.expid}")

# --------------------------------------------------------------------------------
# [LSV-Loc EVALUATION] Load Dataset
# --------------------------------------------------------------------------------

CONFIG = importlib.import_module(f"config.{args.expid}").CONFIG
get_dataset = importlib.import_module(f"utility.Database.{CONFIG.dataloader}").get_dataset

print('==================================================================================')
print('===================== SVR-Loc: Range Image-based Localization ====================')
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

# --------------------------------------------------------------------------------
# [LSV-Loc EVALUATION] Generate Embeddings
# --------------------------------------------------------------------------------

def get_lidar_image_embeddings(valid_loader, model):

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.lidar_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def get_camera_image_embeddings(valid_loader, model):

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.camera_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

# --------------------------------------------------------------------------------
# [LSV-Loc EVALUATION] Evaluation / PnP 
# --------------------------------------------------------------------------------

model_path = CONFIG.final_model_path
model = importlib.import_module(f"utility.Backbone.{CONFIG.model}").Model(CONFIG)
model.to(CONFIG.device)
model.load_state_dict(torch.load(model_path, map_location=CONFIG.device))
model.eval()

valid_dataset = get_dataset(mode=CONFIG.test_dataset, CONFIG=CONFIG)
valid_loader = DataLoader(dataset=valid_dataset, 
                            batch_size=CONFIG.batch_size, 
                            num_workers=CONFIG.num_workers,
                            shuffle=CONFIG.test_shuffle)

# Save or load precomputed embeddings
suffix = CONFIG.test_seq
lidar_file  = os.path.join(CONFIG.expdir, f"lidar_embeddings_Test_{suffix}.pt")
camera_file = os.path.join(CONFIG.expdir, f"camera_embeddings_Test_{suffix}.pt")

lidar_embeddings = None
camera_embeddings = None

if os.path.exists(lidar_file) and os.path.exists(camera_file):
    # load saved embeddings
    lidar_embeddings = torch.load(lidar_file, map_location=CONFIG.device)
    camera_embeddings = torch.load(camera_file, map_location=CONFIG.device)
    print(f'Loaded embeddings: lidar {lidar_embeddings.shape}, camera {camera_embeddings.shape}')
else:
    # [LSV-Loc EVALUATION] Generate Embeddings
    print('===: [LSV-Loc EVALUATION] Embedding GEN: LiDAR')
    lidar_embeddings = get_lidar_image_embeddings(valid_loader, model)
    lidar_embeddings = lidar_embeddings.cuda()

    print('===: [LSV-Loc EVALUATION] Embedding GEN: Camera')
    camera_embeddings = get_camera_image_embeddings(valid_loader, model)
    camera_embeddings = camera_embeddings.cuda()
    
    # save current embeddings to disk
    torch.save(lidar_embeddings.cpu(), lidar_file)
    torch.save(camera_embeddings.cpu(), camera_file)
    # move back to device
    lidar_embeddings = lidar_embeddings.to(CONFIG.device)
    camera_embeddings = camera_embeddings.to(CONFIG.device)
    print(f'Saved embeddings to {lidar_file} and {camera_file}')
    

total_queries = valid_dataset.dbStruct.numQ
eval_tool = evalTool.EvalTool(model, CONFIG.device, patch_size=14)
max_k = 1

ori_disp_list = []
ori_head_list = []
comp_disp_list = []
comp_head_list = []
query_idx_list = []
db_idx_list    = []

# percentile = 95
# for R.R 
percentile = 0

for i in tqdm(range(total_queries), desc="Evaluating queries"):
    
    # ---------------------------------------------------
    # 1. Matching Qeury with Database
    # ---------------------------------------------------
    dot_similarity = lidar_embeddings[i] @ camera_embeddings.T
    values, predictions = torch.topk(dot_similarity.squeeze(0), max_k)

    queryIdx = i
    queryPos = valid_dataset.dbStruct.utmQ[i]

    predictions = predictions.squeeze()
    matches_at_k = {}

    # 1.a) Compute the UTM‐distance between the query and its top‐1 prediction
    dbIdx = predictions.item()  # since max_k = 1
    query_utm = np.array(valid_dataset.dbStruct.utmQ[queryIdx])    # [Easting, Northing]
    db_utm    = np.array(valid_dataset.dbStruct.utmDb[dbIdx])  # [Easting, Northing]
    dist_m    = np.linalg.norm(db_utm[:2] - query_utm[:2])

    # 1.b) if the distance ≥ 25m, skip this pair (we only analyze true positives)
    if dist_m >= 25.0:
        continue
    else: 
        qIdx = i
        db_image_path = eval_tool.process_path(valid_dataset.dbStruct.dbImage[dbIdx])
        query_image_path = eval_tool.process_path(valid_dataset.dbStruct.qImage[qIdx])
        range_image_path = eval_tool.process_path(valid_dataset.dbStruct.qRange[qIdx])

        db_img = cv2.imread(db_image_path)
        db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

        query_img = cv2.imread(query_image_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        range_image = tifffile.imread(range_image_path)
        range_image = range_image/1000

        scale_factor = 1.5
        db_img = cv2.resize(db_img, (int(db_img.shape[1] * scale_factor), int(db_img.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
        query_img = cv2.resize(query_img, (int(query_img.shape[1] * scale_factor), int(query_img.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
        range_image = cv2.resize(range_image, (int(range_image.shape[1] * scale_factor), int(range_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)

        db_img_tensor, grid_size = eval_tool.preprocess_img_tensor(db_img, patch_size=14)
        query_img_tensor, grid_size = eval_tool.preprocess_img_tensor(query_img, patch_size=14)
        
        # ---------------------------------------------------
        # 2. Feature Extract / Attention Map Extract
        # ---------------------------------------------------
        
        img_tensor_DB_Interp = db_img_tensor[0]
        img_tensor_Q_Interp  = query_img_tensor[0]

        with torch.inference_mode():
            img_tensor_DB_Interp = img_tensor_DB_Interp.unsqueeze(0).to(CONFIG.device)
            tokens_DB = model.encoder.encoder.get_intermediate_layers(img_tensor_DB_Interp)[0]
            tokens_DB_CPU = tokens_DB.detach().cpu().numpy()

        del tokens_DB
        gc.collect()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            img_tensor_Q_Interp = img_tensor_Q_Interp.unsqueeze(0).to(CONFIG.device)
            tokens_Q = model.encoder.encoder.get_intermediate_layers(img_tensor_Q_Interp)[0]
            tokens_Q_CPU = tokens_Q.detach().cpu().numpy()

        del tokens_Q
        gc.collect()
        torch.cuda.empty_cache()

        H, W = db_img_tensor.shape[2], db_img_tensor.shape[3]

        cls_map_DB_ORG = eval_tool.attention_map(img_tensor_DB_Interp, layer_idx=-1)
        cls_map_Q_ORG = eval_tool.attention_map(img_tensor_Q_Interp, layer_idx=-1)
        
        # --------------------------------------------------------
        # 3. Feature Append / Attention-based Filtering
        # --------------------------------------------------------
        
        mutual_matches, q_coords, db_coords = eval_tool.mutual_nearest_neighbors(tokens_DB_CPU[0], tokens_Q_CPU[0], grid_size)

        q_scores = cls_map_Q_ORG.flatten()
        d_scores = cls_map_DB_ORG.flatten()

        q_thr = np.percentile(q_scores, percentile)
        d_thr = np.percentile(d_scores, percentile)
    
        filtered_matches = [
            (q, d) for q, d in mutual_matches
            if q_scores[q] >= q_thr and d_scores[d] >= d_thr
        ]

        if len(filtered_matches) < 3:
            print(f"Insufficient filtered matches ({len(filtered_matches)}) for PnP, skipping this pair.")
            continue
        
        # --------------------------------------------------------
        # 4. PnP / Robust Estimator
        # --------------------------------------------------------
        
        from scipy.spatial.transform import Rotation as R

        query_pos = valid_dataset.dbStruct.posQ[qIdx][0:2]
        db_pos = valid_dataset.dbStruct.posDb[dbIdx][0:2]

        query_heading = valid_dataset.dbStruct.posQ[qIdx][3:7]
        db_heading = valid_dataset.dbStruct.posDb[dbIdx][3:7]

        query_rot = R.from_quat([query_heading[1], query_heading[2], query_heading[3], query_heading[0]])
        db_rot = R.from_quat([db_heading[1], db_heading[2], db_heading[3], db_heading[0]])

        query_yaw = query_rot.as_euler('zyx', degrees=True)[0]
        db_yaw = db_rot.as_euler('zyx', degrees=True)[0]

        heading_delta_ori = db_yaw - query_yaw
        delta_ori = db_pos - query_pos

        W_l, H_l = range_image.shape[1], range_image.shape[0]
        cam_height = 0.0

        world_points, image_points = eval_tool.collect_world_and_image_points(filtered_matches, grid_size, range_image, scale_factor=scale_factor, plot=False)

        db_equi = db_img.copy()
        H_l, W_l = db_equi.shape[:2]

        proj_uv = [eval_tool.project_equirectangular(P, W_l, H_l) for P in world_points]
        u_coords = [int(u) for u, v in proj_uv]
        v_coords = [int(v) for u, v in proj_uv]

        # 5) Levenberg–Marquardt 최적화로 pose 추정
        if len(world_points) < 3:
            raise RuntimeError("Not enough correspondences for pose estimation.")

        def fun(params):
            return eval_tool.residuals(params, world_points, image_points, W_l, H_l, cam_height)

        sol = least_squares(
            fun,
            x0=np.zeros(3),
            method='trf',
            loss='huber',
            f_scale=5.0
        )

        tx, ty, yaw = sol.x

        db_utm = valid_dataset.dbStruct.utmDb[dbIdx]
        estimated_query_utm = np.array([db_utm[0] + tx, db_utm[1] + ty])

        db_q = valid_dataset.dbStruct.posDb[dbIdx]
        db_quat = [db_q[3], db_q[4], db_q[5], db_q[6]]
        db_rot = R.from_quat([db_quat[1], db_quat[2], db_quat[3], db_quat[0]])
        db_yaw_rad = db_rot.as_euler('zyx')[0]
        estimated_query_yaw = db_yaw_rad + yaw

        yaw_error = np.rad2deg(estimated_query_yaw) - query_yaw
        if yaw_error > 180:
            yaw_error -= 360
        elif yaw_error < -180:
            yaw_error += 360

        delta_compensated = estimated_query_utm - query_pos
        heading_delta_compensated = yaw_error

        print(f"DB ERROR    // Δ East: {delta_ori[0]:.2f} m, Δ North: {delta_ori[1]:.2f} m, Δ heading: {heading_delta_ori:.2f} deg")
        print(f"COMPR ERROR // Δ East: {delta_compensated[0]:.2f} m, Δ North: {delta_compensated[1]:.2f} m, Δ heading: {heading_delta_compensated:.2f} deg")

        ori_disp_list.append(delta_ori)
        ori_head_list.append(heading_delta_ori)
        comp_disp_list.append(delta_compensated)
        comp_head_list.append(heading_delta_compensated)
        query_idx_list.append(queryIdx)
        db_idx_list.append(dbIdx)

# for문이 끝난 뒤에 파일로 저장
ori_disp_arr  = np.array(ori_disp_list)         # shape: (N, 2)
ori_head_arr  = np.array(ori_head_list)         # shape: (N,)
comp_disp_arr = np.array(comp_disp_list)        # shape: (N, 2)
comp_head_arr = np.array(comp_head_list)        # shape: (N,)
query_idx_arr = np.array(query_idx_list)       # shape: (N,)
db_idx_arr    = np.array(db_idx_list)          # shape: (N,)
print('==================================================================================')
print('===: EVALUATION RESULTS =========================================================')
print('==================================================================================')

save_path_mat = os.path.join(CONFIG.expdir, f"localization_errors_{suffix}.mat")
sio.savemat(save_path_mat, {
    'ori_disp': ori_disp_arr,
    'ori_head': ori_head_arr,
    'comp_disp': comp_disp_arr,
    'comp_head': comp_head_arr,
    'query_idx': query_idx_arr,
    'db_idx': db_idx_arr
})
print(f"Saved localization errors to {save_path_mat}")

print('==================================================================================')
