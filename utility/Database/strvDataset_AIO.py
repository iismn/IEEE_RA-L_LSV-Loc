import os
import cv2
import glob
import numpy as np
import albumentations as A

from PIL import Image
from scipy.io import loadmat
from os.path import join, exists
from collections import namedtuple

import torch
from torch.utils.data import DistributedSampler
import tifffile

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Database Struct Parser
# --------------------------------------------------------------------------------

root_dir = '/home/iris/Workspace/Sangmin/SVR_Loc/dataset/SVR_Dataset_Sync/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to dataset')

struct_dir = join(root_dir, 'MAT/')

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'utmDb', 'qImage', 'qRange', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posQ', 'posDb','posDistSqThr', 'nonTrivPosDistSqThr'])

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] Database Struct Parser
# --------------------------------------------------------------------------------

def parse_dbStruct(path):
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    matStruct = mat['dbStruct']

    dataset = 'strv'
    whichSet = matStruct.whichSet

    dbImage = [f[0] if isinstance(f, (list, np.ndarray)) else f for f in matStruct.dbImageFns]
    dbImage = [f.replace('./SVR_Dataset_Sync', '.').strip() if isinstance(f, str) else f for f in dbImage]
    
    utmDb = matStruct.utmDb.T
    posDb = matStruct.posDb.T

    utmQ = matStruct.utmQ.T
    posQ = matStruct.posQ.T

    qImage = [f[0] if isinstance(f, (list, np.ndarray)) else f for f in matStruct.qImageFns]
    qImage = [f.replace('./SVR_Dataset_Sync', '.').strip() if isinstance(f, str) else f for f in qImage]
    qRange = [f[0] if isinstance(f, (list, np.ndarray)) else f for f in matStruct.qRangeFns]
    qRange = [f.replace('./SVR_Dataset_Sync', '.').strip() if isinstance(f, str) else f for f in qRange]
    
    numDb = matStruct.numImages
    numQ = matStruct.numQueries
    
    posDistThr = matStruct.posDistThr
    posDistSqThr = matStruct.posDistSqThr
    nonTrivPosDistSqThr = matStruct.nonTrivPosDistSqThr

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, qRange, 
            utmQ, numDb, numQ, posDistThr, posQ, posDb,
            posDistSqThr, nonTrivPosDistSqThr)

def get_dataloader(mode, CONFIG):

    suffix = CONFIG.test_seq
    
    if mode=="train":
        struct_file = join(struct_dir, 'SVR_Train_CUT.mat')
    elif mode=="val":
        struct_file = join(struct_dir, f"SVR_Test_{suffix}.mat")
    elif mode=="train_mini":
        struct_file = join(struct_dir, 'SVR_Train_MiniBatch.mat')   
    elif mode=="val_mini":
        struct_file = join(struct_dir, 'SVR_Test_MiniBatch.mat')   

    dataset = CILPDataset_STRV_wPos(
        struct_file=struct_file
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def get_dataset(mode, CONFIG):

    suffix = CONFIG.test_seq

    if mode=="train":
        struct_file = join(struct_dir, 'SVR_Train_CUT.mat')
    elif mode=="val":
        struct_file = join(struct_dir, f"SVR_Test_{suffix}.mat")
    elif mode=="train_mini":
        struct_file = join(struct_dir, 'SVR_Train_MiniBatch.mat')   
    elif mode=="val_mini":
        struct_file = join(struct_dir, 'SVR_Test_MiniBatch.mat')   

    dataset = CILPDataset_STRV_wPos(
        struct_file=struct_file
    )

    return dataset

# --------------------------------------------------------------------------------
# [LSV-Loc TRAINING] CILP Dataset
# --------------------------------------------------------------------------------

class CILPDataset_STRV_wPos(torch.utils.data.Dataset):
    def __init__(self, struct_file, transforms=None, root_dir=root_dir):
        self.root_dir = root_dir
        self.transforms = transforms
        self.dbStruct = parse_dbStruct(struct_file)

        self.db_images = [os.path.join(root_dir, dbIm.strip()) for dbIm in self.dbStruct.dbImage]
        self.query_images = [os.path.join(root_dir, qIm.strip()) for qIm in self.dbStruct.qImage]
        self.range_images = [os.path.join(root_dir, qRng.strip()) for qRng in self.dbStruct.qRange]
        
        self.utm_db = self.dbStruct.utmDb  # shape (numDb, 2)
        self.utm_q  = self.dbStruct.utmQ   # shape (numQ, 2)

        self.pos_db = self.dbStruct.posDb  # shape (numDb, 7)
        self.pos_q  = self.dbStruct.posQ    # shape (numQ, 7)
        
        
        self.length = len(self.db_images)
        if len(self.db_images) != len(self.query_images):
            raise ValueError("Database and Query images should have the same length")
        
    def __getitem__(self, index):
        # --- Database Image (camera) ---
        cam_img = Image.open(self.db_images[index])
        if cam_img.size != (1024, 256):
            cam_img = cam_img.resize((1024, 256))
        if cam_img.mode != "RGB":
            cam_img = cam_img.convert("RGB")
        if self.transforms is not None:
            cam_img = self.transforms(image=np.array(cam_img))['image']
            camera_image = torch.tensor(cam_img).permute(2, 0, 1).float()
        else:
            camera_image = torch.tensor(np.array(cam_img)).permute(2, 0, 1).float()
        
        # --- Query Image (lidar) ---
        query_img = Image.open(self.query_images[index])
        if query_img.mode != "RGB":
            query_img = query_img.convert("RGB")
        if self.transforms is not None:
            query_img = self.transforms(image=np.array(query_img))['image']
            lidar_image = torch.tensor(query_img).permute(2, 0, 1).float()
        else:
            lidar_image = torch.tensor(np.array(query_img)).permute(2, 0, 1).float()

        # --- Range Image ---
        range_image = tifffile.imread(self.range_images[index])
        range_image = range_image.astype(np.float32) / 1000.0  # mm -> m
        range_image = Image.fromarray(range_image)
        if range_image.size != (1024, 256):
            if range_image.size == (1024, 128):
                new_img = Image.new(range_image.mode, (1024, 256))
                new_img.paste(range_image, (0, 64))
                range_image = new_img
            else:
                range_image = range_image.resize((1024, 256))
                
        range_image = torch.tensor(np.array(range_image)).unsqueeze(0).float()
        
        # UTM coords
        utm_db = self.utm_db[index]  # e.g. (x_db, y_db)
        utm_q  = self.utm_q[index]   # e.g. (x_q,  y_q)
        
        pose_db = self.pos_db[index]  # e.g. (x_db, y_db, z_db, qx_db, qy_db, qz_db, qw_db)
        pose_q  = self.pos_q[index]
        
        return {
            'camera_image': camera_image,
            'lidar_image': lidar_image,
            'range_image': range_image,
            'utm_db': torch.tensor(utm_db, dtype=torch.float),
            'utm_q':  torch.tensor(utm_q,  dtype=torch.float),
            'pose_db': torch.tensor(pose_db, dtype=torch.float),
            'pose_q':  torch.tensor(pose_q,  dtype=torch.float),
            'index': index
        }

    def __len__(self):
        return self.length
