import math

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import cv2
import importlib

from dataclasses import dataclass
import tyro

from torch.utils.data import DataLoader, DistributedSampler
import os

##### Global Stuff ######
@dataclass
class Args:
    expid: str = 'strv_eval'
args = tyro.cli(Args)

# --------------------------------------------------------------------------------
# [LSV-Loc EVALUATION] Load Dataset
# --------------------------------------------------------------------------------

CONFIG = importlib.import_module(f"config.{args.expid}").CONFIG
get_dataset = importlib.import_module(f"utility.Database.{CONFIG.dataloader}").get_dataset

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
# [LSV-Loc EVALUATION] Evaluation
# --------------------------------------------------------------------------------

def main():
    
    print('==================================================================================')
    print('===================== SVR-Loc: Range Image-based Localization ====================')
    print('==================================================================================\n')

    print('===: CONFIG =====================================================================')
    for attr in vars(CONFIG):
        if not attr.startswith('__'):  # Skip built-in attributes
            value = getattr(CONFIG, attr)
            value_str = str(value)
            print(f"\033[1;33m{attr.center(30)}: {value_str.center(30)}\033[0m")

    print('===: EVALUATION ==================================================================')
    if CONFIG.resume_model =='best':
        model_path = CONFIG.best_model_path
    elif CONFIG.resume_model =='final':
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
    
    # [LSV-Loc EVALUATION] Generate Embeddings
    print('===: [LSV-Loc EVALUATION] Embedding GEN: LiDAR')
    lidar_embeddings = get_lidar_image_embeddings(valid_loader, model)
    lidar_embeddings = lidar_embeddings.cuda()

    print('===: [LSV-Loc EVALUATION] Embedding GEN: Camera')
    camera_embeddings = get_camera_image_embeddings(valid_loader, model)
    camera_embeddings = camera_embeddings.cuda()

    print('===: [LSV-Loc EVALUATION] Pose GEN')
    num_matches = 0
    total_queries = valid_dataset.dbStruct.numQ

    print('\033[92m===: [LSV-Loc EVALUATION] Evaluation START\033[0m')
    
    recall_Idx = [1,5,10,15,20]
    query_predict = []
    # Initialize recall results dictionary
    recall_results = {k: 0 for k in recall_Idx}
    
    for i, filename in tqdm(enumerate(valid_dataset.dbStruct.qImage)):
        # Get top-k predictions where k is the maximum value in recall_Idx
        max_k = max(recall_Idx)

        dot_similarity = lidar_embeddings[i] @ camera_embeddings.T
        values, predictions = torch.topk(dot_similarity.squeeze(0), max_k)

        queryIdx = i
        queryPos = valid_dataset.dbStruct.utmQ[i]
        
        # Handle multiple predictions (top-k)
        predictions = predictions.squeeze()  # Remove batch dimension if present
        
        # Process each prediction for different k values
        matches_at_k = {}
        
        for pred_idx, prediction in enumerate(predictions):
            predictedPos = valid_dataset.dbStruct.utmDb[prediction]
            distance = math.sqrt((queryPos[0] - predictedPos[0])**2 + 
                    (queryPos[1] - predictedPos[1])**2)
            
            # Update all applicable k values
            for k in recall_Idx:
                if pred_idx < k:  # Only consider predictions within current k
                    if k not in matches_at_k:
                        matches_at_k[k] = distance < CONFIG.threshold_dist
                    else:
                        matches_at_k[k] = matches_at_k[k] or (distance < CONFIG.threshold_dist)
            
        # Update recall counters
        for k in recall_Idx:
            if k in matches_at_k and matches_at_k[k]:
                recall_results[k] += 1
                
        # Collect prediction data for top-1 analysis
        query_predict.append({
            'query_idx': queryIdx,
            'predictions': predictions[:1].cpu().numpy(),  # Just keep top-1 for logging
            'distance': math.sqrt((queryPos[0] - valid_dataset.dbStruct.utmDb[predictions[0]][0])**2 + 
                        (queryPos[1] - valid_dataset.dbStruct.utmDb[predictions[0]][1])**2),
            'matched': matches_at_k.get(1, False)
        })

    # Calculate and print recall for each k
    for k in recall_Idx:
        recall = recall_results[k] / total_queries
        print(f'===: [LSV-Loc EVALUATION] Recall@{k}: {recall*100:.2f}%')
    
    # Save query_predict results to a file
    query_predict_df = pd.DataFrame(query_predict)
    
    if CONFIG.mode == 'LIDAR_simulation':
        output_file = f"{CONFIG.expdir}/query_predict_{CONFIG.LIDAR_name}_{CONFIG.test_seq}.csv"
    else:
        output_file = f"{CONFIG.expdir}/query_predict_{CONFIG.test_seq}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    query_predict_df.to_csv(output_file, index=False)
    
    # Save recall results to a file
    recall_data = [{'k': k, 'recall': recall_results[k] / total_queries * 100} for k in recall_Idx]
    recall_df = pd.DataFrame(recall_data)
    if CONFIG.mode == 'LIDAR_simulation':
        recall_output_file = f"{CONFIG.expdir}/recall_results_{CONFIG.LIDAR_name}_{CONFIG.test_seq}.csv"
    else:
        recall_output_file = f"{CONFIG.expdir}/recall_results_{CONFIG.test_seq}.csv"

    recall_df.to_csv(recall_output_file, index=False)
    
    print(f"===: [LSV-Loc EVALUATION] Saved Predictions / Recall : {output_file} / {recall_output_file}")
    
if __name__ == "__main__":
    main()
