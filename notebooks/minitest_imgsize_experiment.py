#!/usr/bin/env python

"""
A minimal script to compare 300x300 vs 384x384 for DAiSEE-like training 
using optical-flow-based frame selection. We do:

1) Create a tiny subset of Train/Val (so it runs fast).
2) Precompute optical flow frames for each resolution (300, 384).
3) Build LMDB for each resolution.
4) Train for 1 epoch each, track Val Acc / Loss.
5) Print results, then delete ephemeral pkl/LMDB files.

Usage:
  python mini_test.py
"""

import os
import cv2
import gc
import shutil
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import lmdb
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"

# We'll store ephemeral files here:
TEST_CACHE_DIR = BASE_DIR / "cache_mini_test"
TEST_CACHE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We'll test on these 2 resolutions
TEST_RESOLUTIONS = [300, 384]

# We take 50 frames, as your pipeline does
NUM_FRAMES = 50

# We'll do a super short subset
TRAIN_SUBSET_SIZE = 100
VAL_SUBSET_SIZE = 40
EPOCHS = 1

# The 4 target columns in DAiSEE
LABEL_COLS = ["Engagement", "Boredom", "Confusion", "Frustration "]

# ------------------------------------------------
# 1) CREATE TINY SUBSET FOR FAST TEST
# ------------------------------------------------
def create_tiny_subset(train_csv_path, val_csv_path):
    """
    Create a mini subset for train/val with only a few samples,
    store them as DataFrames in memory (no new CSV).
    """
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # random subset
    df_train_small = df_train.sample(n=min(TRAIN_SUBSET_SIZE, len(df_train)), random_state=42).reset_index(drop=True)
    df_val_small   = df_val.sample(n=min(VAL_SUBSET_SIZE, len(df_val)), random_state=42).reset_index(drop=True)

    return df_train_small, df_val_small

# ------------------------------------------------
# 2) SELECT FRAMES (OPTICAL FLOW)
# ------------------------------------------------
def calc_opt_flow_frames(video_folder: Path, num_frames=50):
    # Basic Farneback optical flow approach
    frames = sorted(video_folder.glob("frame_*.jpg"))
    if len(frames) < 2:
        return frames[:num_frames]

    flows = []
    try:
        prev = cv2.imread(str(frames[0]))
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    except:
        return frames[:num_frames]

    for i in range(1, len(frames)):
        try:
            curr = cv2.imread(str(frames[i]))
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            flows.append(np.mean(mag))
            prev_gray = curr_gray
        except:
            flows.append(0.0)

    if len(flows) <= num_frames:
        selected = frames[1:]
    else:
        idx_sorted = np.argsort(flows)[::-1]  # descending
        top_indices = idx_sorted[:num_frames]
        selected = [frames[i+1] for i in top_indices]
    selected.sort(key=lambda x: x.stem)
    return selected[:num_frames]

def get_clip_id(stem):
    # mimic your user function
    stem = stem.strip()
    if stem.startswith("110001"):
        return stem.replace("110001", "202614", 1)
    return stem

# ------------------------------------------------
# 3) CREATE PKL + LMDB (EPHEMERAL)
# ------------------------------------------------
def build_flow_pkl_lmdb(df, split_name, resolution, ephemeral_tag):
    """
    df: subset DataFrame (train or val)
    split_name: "Train" or "Validation"
    ephemeral_tag: unique suffix so we don't clash
    returns path to LMDB
    """
    pkl_name = f"mini_{ephemeral_tag}_{split_name}_{resolution}.pkl"
    lmdb_name = f"mini_{ephemeral_tag}_{split_name}_{resolution}"

    pkl_path = TEST_CACHE_DIR / pkl_name
    lmdb_path = TEST_CACHE_DIR / lmdb_name

    # If pkl or LMDB exist, remove them (fresh start for mini test).
    if pkl_path.exists():
        pkl_path.unlink()
    if lmdb_path.exists():
        # LMDB is a folder with data.mdb, lock.mdb
        shutil.rmtree(lmdb_path, ignore_errors=True)

    # 3a) Build PKL
    valid_indices = []
    frames_list = []
    for idx, row in df.iterrows():
        clip_id = get_clip_id(row["ClipID"].split('.')[0])
        folder = FRAMES_DIR / split_name / clip_id
        if folder.exists():
            selected_frames = calc_opt_flow_frames(folder, NUM_FRAMES)
            if selected_frames:
                valid_indices.append(idx)
                frames_list.append(selected_frames)
    # Save
    data = {"valid_indices": valid_indices, "frame_paths": frames_list}
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    # 3b) Build LMDB from PKL
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    env = lmdb.open(str(lmdb_path), map_size=1 * 1024**3)
    # minimal backbone (for memory/time sake) => let's do a small timm net
    # For consistency, let's still do efficientnetv2_l but you can pick smaller if you want faster
    backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    # ^ example smaller model - or you can do timm efficientnet
    backbone.eval().to(DEVICE)
    for param in backbone.parameters():
        param.requires_grad = False

    with env.begin(write=True) as txn:
        for arr_idx, fp_list in tqdm(enumerate(frames_list), total=len(frames_list), desc=f"LMDB {split_name} {resolution}"):
            video_features = []
            for fp in fp_list:
                try:
                    img = Image.open(fp).convert('RGB')
                except:
                    img = Image.new('RGB', (resolution, resolution))
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    feat = backbone(tensor)
                feat = feat.squeeze(0).cpu().half().numpy()
                video_features.append(feat)
            # store
            key = f"vid_{valid_indices[arr_idx]}".encode("utf-8")
            txn.put(key, pickle.dumps(np.stack(video_features)))
    env.close()

    return pkl_path, lmdb_path

# ------------------------------------------------
# 4) A Minimal LMDB Dataset
# ------------------------------------------------
class MiniFlowDataset(Dataset):
    def __init__(self, df, split_name, pkl_path, lmdb_path):
        super().__init__()
        self.df = df
        self.split_name = split_name
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.valid_indices = data["valid_indices"]
        self.frame_paths = data["frame_paths"]
        self.df_small = self.df.iloc[self.valid_indices].reset_index(drop=True)

        self.lmdb_env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.df_small)

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = f"vid_{self.valid_indices[idx]}".encode("utf-8")
            data_bytes = txn.get(key)
            feats = pickle.loads(data_bytes)  # shape (T, some_dim)
        # we just flatten or keep it as is
        feats = torch.from_numpy(feats)  # (T, D)
        label_row = self.df_small.iloc[idx][LABEL_COLS].astype(int).values
        labels = torch.tensor(label_row, dtype=torch.long)
        return feats, labels

# ------------------------------------------------
# 5) A Minimal Model
# ------------------------------------------------
class SimpleClassifier(nn.Module):
    """
    We'll do a minimal approach:
    - Input shape: (B,T, D) where D is ResNet18's output (~1000).
    - We'll do a 1D conv or a small GRU -> final linear => 4 dims * 4 classes = 16
    """
    def __init__(self, input_dim=1000, hidden=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 16)

    def forward(self, x):
        """
        x: (B,T,D)
        """
        # GRU
        out, h_n = self.gru(x)
        # h_n: (num_layers=1, B, hidden)
        # we can take h_n[-1]
        final = h_n[-1]  # shape (B, hidden)
        logits = self.fc(final)  # (B,16)
        return logits.view(-1,4,4)

# ------------------------------------------------
# 6) Minimal Train/Val for 1 epoch
# ------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_samples = 0

    for feats, labels in loader:
        feats = feats.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(feats)  # (B,4,4)
            loss = 0
            # basic multi-class CE for each dimension
            for d in range(4):
                loss += nn.functional.cross_entropy(outputs[:,d,:], labels[:,d])
            loss = loss / 4.0
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        total_samples += feats.size(0)

    return total_loss / total_samples

def validate(model, loader):
    model.eval()
    total_loss = 0
    total_samples = 0

    all_preds = []
    all_gts   = []
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for feats, labels in loader:
            feats = feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(feats)
            # multi-dim CE
            loss = 0
            for d in range(4):
                loss += nn.functional.cross_entropy(outputs[:,d,:], labels[:,d])
            loss = loss / 4.0
            total_loss += loss.item() * feats.size(0)
            total_samples += feats.size(0)
            # check predictions
            preds = torch.argmax(outputs, dim=2)  # (B,4)
            all_preds.append(preds.cpu().numpy())
            all_gts.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts, axis=0)

    # let's do a very naive average accuracy across the 4 dims
    correct = (all_preds == all_gts).sum()
    total = all_preds.size
    acc = correct / total
    return avg_loss, acc

# ------------------------------------------------
# 7) End-to-End test
# ------------------------------------------------
def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv   = LABELS_DIR / "ValidationLabels.csv"

    # Step A: subset
    df_train_small, df_val_small = create_tiny_subset(train_csv, val_csv)

    results = []

    for res in TEST_RESOLUTIONS:
        ephemeral_tag = f"res{res}"
        # Build ephemeral PKL + LMDB for train
        train_pkl, train_lmdb = build_flow_pkl_lmdb(df_train_small, "Train", res, ephemeral_tag+"_train")
        val_pkl,   val_lmdb   = build_flow_pkl_lmdb(df_val_small,   "Validation", res, ephemeral_tag+"_val")

        train_ds = MiniFlowDataset(df_train_small, "Train", train_pkl, train_lmdb)
        val_ds   = MiniFlowDataset(df_val_small,   "Validation", val_pkl, val_lmdb)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

        # Build model
        model = SimpleClassifier(input_dim=1000, hidden=128).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # short training: 1 epoch
        print(f"\n--- Training @ res={res} for {EPOCHS} epoch(s) ---")
        for epoch in range(EPOCHS):
            tr_loss = train_one_epoch(model, train_loader, optimizer, None)
            val_loss, val_acc = validate(model, val_loader)
            print(f" Epoch {epoch+1}/{EPOCHS} => Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        results.append((res, val_loss, val_acc))

        # Delete ephemeral files
        print(f"Deleting ephemeral files for res={res} ...")
        if os.path.exists(train_pkl):
            os.remove(train_pkl)
        if os.path.exists(val_pkl):
            os.remove(val_pkl)
        if os.path.exists(train_lmdb):
            shutil.rmtree(train_lmdb, ignore_errors=True)
        if os.path.exists(val_lmdb):
            shutil.rmtree(val_lmdb, ignore_errors=True)

    print("\n=== COMPARISON RESULTS ===")
    for (r, vl, va) in results:
        print(f"Resolution={r} => ValLoss={vl:.4f}, ValAcc={va:.4f}")

if __name__ == "__main__":
    main()
