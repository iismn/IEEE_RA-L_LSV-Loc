# -*- coding: utf-8 -*-
"""
Cross-modal Representation Check for SVR-Loc
- Positive vs Negative cosine similarity distributions
- Joint t-SNE (Color by Modality / Color by Place-ID)

Requirements:
  pip install numpy scipy torch scikit-learn matplotlib

Usage:
  python analyze_repr.py
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.manifold import TSNE
import csv

# =========================
# Paths (edit if needed)
# =========================
CAM_EMB_PATH = "/home/iris/Workspace/Sangmin/SVR_Loc/result/04160633_dinov2_vitb14_reg/camera_embeddings_Test_ComplexUrban05.pt"
LID_EMB_PATH = "/home/iris/Workspace/Sangmin/SVR_Loc/result/04160633_dinov2_vitb14_reg/lidar_embeddings_Test_ComplexUrban05.pt"
MAT_META_PATH = "/home/iris/Workspace/Sangmin/SVR_Loc/dataset/SVR_Dataset_Sync/MAT/SVR_Test_ComplexUrban05.mat"

# Output dir (default: alongside camera embedding dir)
OUT_DIR = os.path.join(os.path.dirname(CAM_EMB_PATH), "analysis_repr")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Configs
# =========================
POS_RADIUS_M = 25.0       # Positive threshold (same place)
NEG_RADIUS_MIN_M = 100.0  # Negative threshold minimum distance
MAX_TSNE_PAIRS = 200      # Pairs used for t-SNE (pair = 1 LiDAR + 1 StreetView)
PERPLEXITY = 30
TSNE_ITER = 2000
SEED = 42                 # for reproducibility in sampling & t-SNE

random.seed(SEED)
np.random.seed(SEED)

# =========================
# Utilities
# =========================
def safe_load_embeddings(pt_path):
    embs = torch.load(pt_path, map_location="cpu")
    # Accept both tensor and dict-like
    if isinstance(embs, dict):
        # try common keys
        for k in ["embeddings", "image_embeddings", "features", "feat"]:
            if k in embs:
                embs = embs[k]
                break
        # if still dict-like keys are ambiguous, try first tensor value
        if isinstance(embs, dict):
            for v in embs.values():
                if torch.is_tensor(v):
                    embs = v
                    break
    if not torch.is_tensor(embs):
        raise ValueError(f"Cannot parse embeddings from {pt_path}")
    return embs.detach().cpu()

# ...existing code...
def extract_utm_arrays(mat_path):
    """
    MAT 구조에서 utmQ, utmDb를 꺼내 (N,2) 형태로 반환합니다.
    """
    def _ensure_n2(arr):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"UTM array must be 2-D, got {arr.shape}")
        if arr.shape[1] == 2:
            return arr
        if arr.shape[0] == 2:
            return arr.T
        raise ValueError(f"Unexpected UTM shape {arr.shape}")

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if "utmQ" in mat and "utmDb" in mat:
        return _ensure_n2(mat["utmQ"]), _ensure_n2(mat["utmDb"])

    if "dbStruct" in mat:
        ds = mat["dbStruct"]
        try:
            utmQ = _ensure_n2(ds.utmQ.T)
            utmDb = _ensure_n2(ds.utmDb.T)
            return utmQ, utmDb
        except AttributeError:
            utmQ = _ensure_n2(ds["utmQ"].T)
            utmDb = _ensure_n2(ds["utmDb"].T)
            return utmQ, utmDb

    raise ValueError(f"Cannot find 'utmQ'/'utmDb' in {mat_path}.")
# ...existing code...

def l2_dist(a, b):
    # a: [N,2], b: [M,2]
    # returns [N,M]
    return np.sqrt(((a[:, None, :] - b[None, :, :])**2).sum(axis=2))

def cosine_sim(a, b):
    # a,b: 1D tensors
    return F.cosine_similarity(a, b, dim=0).item()

# =========================
# Load data
# =========================
print("[Load] Embeddings...")
camE = safe_load_embeddings(CAM_EMB_PATH)   # [M, D] - Street View (DB)
lidE = safe_load_embeddings(LID_EMB_PATH)   # [N, D] - LiDAR (Query)
print(f"  Camera embeddings: {tuple(camE.shape)}")
print(f"  LiDAR  embeddings: {tuple(lidE.shape)}")

print("[Load] UTM coordinates from MAT...")
utmQ, utmDb = extract_utm_arrays(MAT_META_PATH)
print(f"  utmQ shape: {tuple(utmQ.shape)} | utmDb shape: {tuple(utmDb.shape)}")

assert lidE.shape[0] == utmQ.shape[0], "LiDAR embedding count must match utmQ count"
assert camE.shape[0] == utmDb.shape[0], "Camera embedding count must match utmDb count"

# (Optional) normalize embeddings (often already L2-normalized; safe to do again)
lidE = torch.nn.functional.normalize(lidE, dim=1)
camE = torch.nn.functional.normalize(camE, dim=1)

# =========================
# Positive / Negative selection
# =========================
print("[Compute] Distance matrix & Pairing...")
# To avoid a full N*M when huge, do it in chunks if needed; otherwise compute directly:
N = utmQ.shape[0]
M = utmDb.shape[0]

# Full distance matrix (if memory allows); otherwise chunk
CHUNK = 2000
if N * M <= 4_000_000:  # heuristic threshold
    D = l2_dist(utmQ, utmDb)  # [N, M]
else:
    print("  Using chunked distance computation...")
    D = np.empty((N, M), dtype=np.float32)
    for s in range(0, N, CHUNK):
        e = min(s+CHUNK, N)
        D[s:e] = l2_dist(utmQ[s:e], utmDb)

# For each query i, choose:
#   Positive: among DB with dist < 25m, pick nearest. If none, skip.
#   Negative: among DB with dist > 100m, pick one random (or farthest).
pos_pairs = []
neg_pairs = []

for i in range(N):
    row = D[i]
    pos_idx = np.where(row < POS_RADIUS_M)[0]
    if pos_idx.size == 0:
        continue
    # nearest positive
    j_pos = pos_idx[row[pos_idx].argmin()]
    pos_pairs.append((i, j_pos))

    neg_idx = np.where(row > NEG_RADIUS_MIN_M)[0]
    if neg_idx.size > 0:
        j_neg = int(np.random.choice(neg_idx))
        neg_pairs.append((i, j_neg))

print(f"  #Pos pairs: {len(pos_pairs)} | #Neg pairs: {len(neg_pairs)}")

# =========================
# Cosine similarity distributions
# =========================
print("[Compute] Cosine similarity distributions...")
pos_sims = []
for (qi, dj) in pos_pairs:
    pos_sims.append(cosine_sim(lidE[qi], camE[dj]))

neg_sims = []
for (qi, dj) in neg_pairs:
    neg_sims.append(cosine_sim(lidE[qi], camE[dj]))

pos_sims = np.array(pos_sims, dtype=np.float32)
neg_sims = np.array(neg_sims, dtype=np.float32)

# Save CSV summary
csv_path = os.path.join(OUT_DIR, "cosine_similarity_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    def stats(x):
        return [len(x), float(np.mean(x)), float(np.std(x)), float(np.min(x)),
                float(np.percentile(x,25)), float(np.median(x)), float(np.percentile(x,75)), float(np.max(x))]
    w.writerow(["positive"] + stats(pos_sims))
    w.writerow(["negative"] + stats(neg_sims))
print(f"[Save] CSV => {csv_path}")

# Plot histogram
plt.figure(figsize=(7,5))
plt.hist(pos_sims, bins=50, alpha=0.6, label=f'Positive (N={len(pos_sims)})')
plt.hist(neg_sims, bins=50, alpha=0.6, label=f'Negative (N={len(neg_sims)})')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Cross-modal Cosine Similarity Distributions')
plt.legend()
plt.tight_layout()
fig_hist_path = os.path.join(OUT_DIR, "cosine_distributions.png")
plt.savefig(fig_hist_path, dpi=200)
plt.close()
print(f"[Save] Figure => {fig_hist_path}")

# =========================
# t-SNE (small-scale, optional but useful for the review)
# =========================
print("[t-SNE] Sampling up to pairs =", MAX_TSNE_PAIRS)
pairs = pos_pairs.copy()
if len(pairs) > MAX_TSNE_PAIRS:
    random.shuffle(pairs)
    pairs = pairs[:MAX_TSNE_PAIRS]

# Build Z (2 vectors per pair), labels: modality & place-id
Z = []
mod_labels = []   # 0=LiDAR, 1=StreetView
place_labels = [] # same pair => same ID

for pid, (qi, dj) in enumerate(pairs):
    Z.append(lidE[qi].numpy());  mod_labels.append(0); place_labels.append(pid)
    Z.append(camE[dj].numpy());  mod_labels.append(1); place_labels.append(pid)

Z = np.stack(Z, axis=0)
print(f"  t-SNE input embeddings: {Z.shape}")

tsne = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    init='pca',
    learning_rate='auto',
    n_iter=TSNE_ITER,
    random_state=SEED,
    verbose=1
)
Z2 = tsne.fit_transform(Z)  # [2*P, 2]

# Plot-1: color by modality
plt.figure(figsize=(6,5))
mask_l = (np.array(mod_labels) == 0)
mask_c = (np.array(mod_labels) == 1)
plt.scatter(Z2[mask_l,0], Z2[mask_l,1], s=12, alpha=0.75, label='LiDAR')
plt.scatter(Z2[mask_c,0], Z2[mask_c,1], s=12, alpha=0.75, label='Street View')
plt.title('t-SNE of Joint Embeddings (Color by Modality)')
plt.legend()
plt.tight_layout()
tsne_mod_path = os.path.join(OUT_DIR, "tsne_modality.png")
plt.savefig(tsne_mod_path, dpi=200)
plt.close()
print(f"[Save] Figure => {tsne_mod_path}")

# Plot-2: color by place-id (paired points near each other)
plt.figure(figsize=(6,5))
sc = plt.scatter(Z2[:,0], Z2[:,1], c=np.array(place_labels), s=12, alpha=0.85, cmap='tab20')
plt.title('t-SNE of Joint Embeddings (Color by Place-ID)')
plt.colorbar(sc, fraction=0.046, pad=0.04)
plt.tight_layout()
tsne_place_path = os.path.join(OUT_DIR, "tsne_placeid.png")
plt.savefig(tsne_place_path, dpi=200)
plt.close()
print(f"[Save] Figure => {tsne_place_path}")

print("\n[Done]")
print(f"Outputs saved in: {OUT_DIR}")
