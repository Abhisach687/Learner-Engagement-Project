#!/usr/bin/env python

import os
import cv2
import gc
import time
import random
import pickle
import lmdb
import sqlite3
import logging
import numpy as np
import optuna
import timm
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from timm.models.layers import DropPath
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from optuna.pruners import MedianPruner

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------
# GLOBALS & HYPERPARAMETERS
# -----------------------------------------------------
GRADIENT_ACCUM_STEPS = 4
NUM_FRAMES = 50
PROG_SCHEDULE = [(112, 5), (224, 10), (300, 5)]
LABEL_SMOOTHING = 0.1
EARLYSTOP_PATIENCE = 8

# Final short raw pass parameters
FINAL_FINE_EPOCHS = 3
FINAL_FINE_LR = 1e-5

# LMDB map size ~1GB
LMDB_MAP_SIZE = 1 * (1024**3)

# For LMDB generation – set SAMPLE_RATIO to 1.0 for full run (or <1.0 for debugging)
SAMPLE_RATIO = 1.0
BATCH_SIZE = 16  # For LMDB generation (batch-based GPU extraction)

# -----------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------
# Helper: guess_split
# -----------------------------------------
def guess_split(csv_path: Path) -> str:
    """Determine dataset split from CSV filename."""
    name = csv_path.stem.lower()
    if 'train' in name:
        return 'Train'
    if 'val' in name:
        return 'Validation'
    if 'test' in name:
        return 'Test'
    return 'Train'

# -----------------------------------------
# Weighted Ordinal Cross-Entropy Loss & Class Weights
# -----------------------------------------
def compute_class_weights_for_daisee(train_csv, val_csv):
    label_names = ["Engagement", "Boredom", "Confusion", "Frustration"]
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df_train.columns = df_train.columns.str.strip()
    df_val.columns   = df_val.columns.str.strip()
    combined = pd.concat([df_train, df_val], ignore_index=True)
    combined.columns = combined.columns.str.strip()
    weights_dict = {}
    for lbl in label_names:
        arr = combined[lbl].astype(int).values
        w = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=arr)
        weights_dict[lbl] = torch.tensor(w, dtype=torch.float32)
    return weights_dict

class WeightedOrdinalCrossEntropyLoss(nn.Module):
    label_names = ["Engagement", "Boredom", "Confusion", "Frustration"]
    def __init__(self, smoothing=0.1, weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        # logits: (B,4,4), targets: (B,4)
        total = 0.0
        for d, name in enumerate(self.label_names):
            dim_logits = logits[:, d, :]
            dim_targets = targets[:, d]
            bin_labels = torch.zeros_like(dim_logits)
            for c in range(4):
                bin_labels[:, c] = (dim_targets > c).float()
            if self.smoothing > 0:
                bin_labels = bin_labels * (1 - self.smoothing) + 0.5 * self.smoothing
            bce_loss = self.bce(dim_logits, bin_labels)
            if self.weights is not None and name in self.weights:
                w_arr = self.weights[name].to(logits.device)
                sample_w = w_arr[dim_targets.long()]
                bce_loss = bce_loss * sample_w.unsqueeze(1)
            total += bce_loss.mean()
        return total / 4.0

# -----------------------------------------
# LMDB Generation Functions (RGB & Flow)
# -----------------------------------------
def get_csv_clip_id(st):
    c = st.strip()
    if c.startswith("110001"):
        return c.replace("110001", "202614", 1)
    return c

def _uniform_select(folder: Path, num_frames=50):
    frames = sorted(folder.glob("frame_*.jpg"))
    total = len(frames)
    if total < 1:
        return []
    if total <= num_frames:
        return frames
    idxes = np.linspace(0, total - 1, num_frames, dtype=int)
    return [frames[i] for i in idxes]

def precompute_uniform_frames(csv_file: Path, video_root: Path, num_frames=50, resolution=224, flow=False):
    df = pd.read_csv(csv_file, dtype=str)
    df.columns = df.columns.str.strip()
    split = csv_file.stem.replace("Labels", "").strip()
    tag = "flow" if flow else "rgb"
    cache_path = CACHE_DIR / f"precomputed_{csv_file.stem}_{tag}_frame_{num_frames}_{resolution}.pkl"
    if cache_path.exists():
        print(f"[Precompute] found => {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    valid_idx = []
    precomp = []
    skip = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"[PrecomputeUniform] {csv_file.stem} {tag} {resolution}"):
        clip_id = get_csv_clip_id(row["ClipID"].split('.')[0])
        folder = video_root / split / clip_id
        if not folder.exists():
            skip += 1
            continue
        frames = _uniform_select(folder, num_frames)
        if len(frames) >= num_frames:
            precomp.append(frames[:num_frames])
            valid_idx.append(i)
        else:
            skip += 1
    c = {"valid_indices": valid_idx, "precomputed_frames": precomp}
    with open(cache_path, "wb") as f:
        pickle.dump(c, f)
    print(f"[Precompute] skip={skip}/{len(df)} => {cache_path}")
    return c

def convert_to_lmdb(csv_file: Path, num_frames=50, resolution=224,
                    rgb=True, transform=None, lmdb_map_size=2*(1024**3)):
    tag = "rgb" if rgb else "flow"
    if transform is None:
        if rgb:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])
            ])
    lmdb_path = CACHE_DIR / f"lmdb_{tag}_{csv_file.stem}_{resolution}"
    if (lmdb_path / "data.mdb").exists():
        print(f"[LMDB] => found {lmdb_path}")
        return lmdb_path
    pkl_path = CACHE_DIR / f"precomputed_{csv_file.stem}_{tag}_frame_{num_frames}_{resolution}.pkl"
    if not pkl_path.exists():
        _ = precompute_uniform_frames(csv_file, FRAMES_DIR, num_frames, resolution, flow=(not rgb))
    with open(pkl_path, "rb") as f:
        c = pickle.load(f)
    valid_idx = c["valid_indices"]
    frames_list = c["precomputed_frames"]
    in_ch = 3 if rgb else 1
    backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True, in_chans=in_ch)
    backbone.reset_classifier(0)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    env = lmdb.open(str(lmdb_path), map_size=lmdb_map_size)
    print(f"[LMDB] => building {lmdb_path}")
    with env.begin(write=True) as txn:
        for i, flist in tqdm(enumerate(frames_list), total=len(frames_list),
                             desc=f"LMDB building {tag} {csv_file.stem}@{resolution}"):
            feats = []
            n = len(flist)
            nbatches = (n + BATCH_SIZE - 1) // BATCH_SIZE
            for b_idx in range(nbatches):
                batch_files = flist[b_idx*BATCH_SIZE : (b_idx+1)*BATCH_SIZE]
                batch_imgs = []
                for fp in batch_files:
                    try:
                        img = Image.open(fp)
                        if rgb:
                            img = img.convert("RGB")
                        else:
                            img = img.convert("L")
                    except:
                        if rgb:
                            img = Image.new("RGB", (resolution, resolution))
                        else:
                            img = Image.new("L", (resolution, resolution))
                    batch_imgs.append(transform(img).unsqueeze(0).to(device))
                if not batch_imgs:
                    continue
                batch_tensor = torch.cat(batch_imgs, dim=0)
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = backbone(batch_tensor)
                out = out.cpu().half().detach().numpy()
                feats.append(out)
            if feats:
                final_arr = np.concatenate(feats, axis=0)  # (NUM_FRAMES, feat_dim)
                key = f"video_{valid_idx[i]}".encode("utf-8")
                txn.put(key, pickle.dumps(final_arr))
    env.close()
    return lmdb_path

# -----------------------------------------
# Hybrid LMDB Dataset
# -----------------------------------------
class HybridLMDBDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, rgb_lmdb, flow_lmdb, resolution=224, num_frames=50):
        df = pd.read_csv(csv_file, dtype=str)
        df.columns = df.columns.str.strip()
        rgb_pkl = CACHE_DIR / f"precomputed_{csv_file.stem}_rgb_frame_{num_frames}_{resolution}.pkl"
        flow_pkl = CACHE_DIR / f"precomputed_{csv_file.stem}_flow_frame_{num_frames}_{resolution}.pkl"
        with open(rgb_pkl, "rb") as f:
            c_rgb = pickle.load(f)
        with open(flow_pkl, "rb") as f2:
            c_flow = pickle.load(f2)
        rgb_idx = c_rgb["valid_indices"]
        flow_idx = c_flow["valid_indices"]
        inter = list(set(rgb_idx).intersection(set(flow_idx)))
        inter.sort()
        self.valid_idx = inter
        self.df_small = df.iloc[self.valid_idx].reset_index(drop=True)
        self.rgb_lmdb = str(rgb_lmdb)
        self.flow_lmdb = str(flow_lmdb)
        self.env_rgb = None
        self.env_flow = None
        self.num_frames = num_frames

    def _init_rgb_env(self):
        if self.env_rgb is None:
            self.env_rgb = lmdb.open(self.rgb_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env_rgb

    def _init_flow_env(self):
        if self.env_flow is None:
            self.env_flow = lmdb.open(self.flow_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env_flow

    def __len__(self):
        return len(self.df_small)

    def __getitem__(self, idx):
        ridx = self.valid_idx[idx]
        k = f"video_{ridx}".encode("utf-8")
        rgb_env = self._init_rgb_env()
        flow_env = self._init_flow_env()
        with rgb_env.begin(write=False) as txn:
            rgb_bytes = txn.get(k)
            if rgb_bytes is None:
                raise KeyError(f"missing => {k} in rgb LMDB")
            rgb_feats_np = pickle.loads(rgb_bytes)
        with flow_env.begin(write=False) as txn2:
            flow_bytes = txn2.get(k)
            if flow_bytes is None:
                raise KeyError(f"missing => {k} in flow LMDB")
            flow_feats_np = pickle.loads(flow_bytes)
        # Convert LMDB features from FP16 to FP32 here
        rgb_feats = torch.from_numpy(rgb_feats_np).float()
        flow_feats = torch.from_numpy(flow_feats_np).float()

        fused = torch.cat([rgb_feats, flow_feats], dim=1)  # (T, 2*feat_dim)
        row = self.df_small.iloc[idx]
        labs = row[["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int).values
        return fused, torch.tensor(labs, dtype=torch.long)

# -----------------------------------------
# Final Raw-Only Dataset (for Final Finetuning)
# -----------------------------------------
class VideoDatasetRawOnly(torch.utils.data.Dataset):
    def __init__(self, csvfile, root, n_frames=50, transform=None):
        df = pd.read_csv(csvfile, dtype=str)
        df.columns = df.columns.str.strip()
        self.df = df
        self.transform = transform
        pklp = CACHE_DIR / f"precomp_{csvfile.stem}_raw_{n_frames}_300.pkl"
        if not pklp.exists():
            data = []
            v_idx = []
            sp = csvfile.stem.replace("Labels", "").strip()
            skip = 0
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"RawOnly {csvfile.stem}"):
                clip_id = row["ClipID"].split('.')[0]
                clip_id = get_csv_clip_id(clip_id)
                fold = root / sp / clip_id
                if fold.exists():
                    frames = _uniform_select(fold, n_frames)
                    if len(frames) >= n_frames:
                        data.append(frames[:n_frames])
                        v_idx.append(i)
                    else:
                        skip += 1
                else:
                    skip += 1
            c = {"valid_indices": v_idx, "frames": data}
            with open(pklp, "wb") as f:
                pickle.dump(c, f)
        else:
            with open(pklp, "rb") as f:
                c = pickle.load(f)
        self.df_small = df.iloc[c["valid_indices"]].reset_index(drop=True)
        self.frames = c["frames"]
        self.transform = transform
    def __len__(self):
        return len(self.df_small)
    def __getitem__(self, idx):
        pths = self.frames[idx]
        seq = []
        for fp in pths:
            try:
                im = Image.open(fp).convert("RGB")
            except:
                im = Image.new("RGB", (300,300))
            if self.transform:
                im = self.transform(im)
            seq.append(im)
        vid = torch.stack(seq, dim=0)
        row = self.df_small.iloc[idx]
        labs = row[["Engagement","Boredom","Confusion","Frustration"]].astype(int).values
        return vid, torch.tensor(labs, dtype=torch.long)

# -----------------------------------------
# Model Architecture: Dual Stream LMDB Model
# -----------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

class CrossAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, sp, tp):
        q = self.query(sp).unsqueeze(1)
        k = self.key(tp).unsqueeze(2)
        attn = self.softmax(torch.bmm(q, k))
        val = self.value(tp)
        return attn.squeeze(-1) * val

class DualStreamLMDBModel(nn.Module):
    """
    Takes fused LMDB features of shape (B, T, 2*feat_dim) and passes through:
    CBAM -> BiLSTM -> CrossAttention -> Classifier
    """
    def __init__(self, feat_dim=2560, lstm_hidden=256, lstm_layers=1, dropout_rate=0.5, classifier_hidden=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.cbam = CBAM(feat_dim, 16, 7)
        self.bilstm = nn.LSTM(feat_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.spatial_proj = nn.Linear(feat_dim, classifier_hidden)
        self.temporal_proj = nn.Linear(2 * lstm_hidden, classifier_hidden)
        self.cross_attn = CrossAttention(classifier_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(classifier_hidden * 2, 16)
    def forward(self, x):
        # x: (B, T, 2*feat_dim)
        B, T, F = x.shape
        x = x.permute(0, 2, 1)
        x = self.cbam(x)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.bilstm(x)
        temporal_ctx = torch.cat([h_n[-2], h_n[-1]], dim=1)
        spatial_feat = x.mean(dim=1)
        sp = self.spatial_proj(spatial_feat)
        tp = self.temporal_proj(temporal_ctx)
        cross = self.cross_attn(sp, tp)
        cat = torch.cat([sp, cross], dim=1)
        cat = self.dropout(cat)
        logits = self.classifier(cat)
        return logits.view(B, 4, 4)

def mixup_in_feat_space(x, labs, alpha=0.2):
    if alpha <= 0:
        return x, None
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    x_m = lam * x + (1 - lam) * x[idx, :]
    dims = []
    for d in range(4):
        oh = nn.functional.one_hot(labs[:, d], num_classes=4).float()
        dims.append(oh)
    labs_oh = torch.stack(dims, dim=1)
    labs_oh_m = lam * labs_oh + (1 - lam) * labs_oh[idx, :]
    return x_m, labs_oh_m

# -----------------------------------------
# Progressive Training Function
# -----------------------------------------
def progressive_train_dual_lmdb(
    model, total_epochs, lr, ckpt_path, batch_size,
    train_csv, val_csv, schedule=PROG_SCHEDULE,
    patience=EARLYSTOP_PATIENCE,
    gradient_accum_steps=GRADIENT_ACCUM_STEPS,
    weights_dict=None,
    mixup_alpha=0.2,
    ema_decay=0.995
):
    """
    Progressive training over specified resolutions.
    LR is passed from Optuna and not overridden.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = WeightedOrdinalCrossEntropyLoss(LABEL_SMOOTHING, weights=weights_dict).to(device)
    # EMA model
    ema_model = AveragedModel(model, avg_fn=lambda avg, p, n: ema_decay * avg + (1 - ema_decay) * p)
    scaler = GradScaler()
    best_val = float('inf')
    es_count = 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    total_scheduled = sum(e for _, e in schedule)
    current_epoch = 0

    for (res, ep) in schedule:
        # Build LMDB for both RGB and Flow
        flow_lmdb = convert_to_lmdb(train_csv, NUM_FRAMES, res, rgb=False, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
        flow_val = convert_to_lmdb(val_csv, NUM_FRAMES, res, rgb=False, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
        rgb_lmdb = convert_to_lmdb(train_csv, NUM_FRAMES, res, rgb=True, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
        rgb_val = convert_to_lmdb(val_csv, NUM_FRAMES, res, rgb=True, transform=None, lmdb_map_size=LMDB_MAP_SIZE)

        train_ds = HybridLMDBDataset(train_csv, rgb_lmdb, flow_lmdb, res, NUM_FRAMES)
        val_ds = HybridLMDBDataset(val_csv, rgb_val, flow_val, res, NUM_FRAMES)

        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for e in range(ep):
            current_epoch += 1
            print(f"[Progressive-dualLMDB+EMA] => E{current_epoch}/{total_scheduled}, res={res}, lr={lr}")
            model.train()
            run_loss = 0.0
            for i, (feats, labs) in enumerate(tqdm(train_ld, desc="Train-dualLMDB")):
                feats = feats.to(device)
                labs = labs.to(device)
                feats_m, labs_oh = mixup_in_feat_space(feats, labs, mixup_alpha)
                with autocast(device_type='cuda', dtype=torch.float16):
                    if labs_oh is not None:
                        outs = model(feats_m)
                        loss = criterion(outs, labs_oh.argmax(dim=2))
                    else:
                        outs = model(feats)
                        loss = criterion(outs, labs)
                scaler.scale(loss / gradient_accum_steps).backward()
                if (i + 1) % gradient_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.zero_grad()
                    # Update EMA if not in last 5 epochs
                    if current_epoch < (total_scheduled - 5):
                        ema_model.update_parameters(model)
                run_loss += loss.item() * feats.size(0)
                del feats, labs, outs, loss
            train_loss = run_loss / len(train_ld.dataset)

            # Use EMA model for validation if not in final 5 epochs
            if current_epoch < (total_scheduled - 5):
                eval_model = ema_model
            else:
                eval_model = model

            eval_model.eval()
            val_loss = 0.0
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                for feats, labs in val_ld:
                    feats = feats.to(device)
                    labs = labs.to(device)
                    out = eval_model(feats)
                    vls = criterion(out, labs)
                    val_loss += vls.item() * feats.size(0)
            val_loss /= len(val_ld.dataset)

            print(f" => E{current_epoch}: train={train_loss:.4f}, val(EMA)={val_loss:.4f}")
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                st_dict = ema_model.module.state_dict() if current_epoch < (total_scheduled - 5) else model.state_dict()
                st = {
                    "epoch": current_epoch,
                    "val_loss": best_val,
                    "model_state_dict": st_dict,
                    "optimizer_state_dict": optimizer.state_dict()
                }
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save(st, tmp)
                if ckpt_path.exists():
                    ckpt_path.unlink()
                tmp.rename(ckpt_path)
                es_count = 0
            else:
                es_count += 1
            if es_count >= patience:
                print(f"[EarlyStop-dualLMDB+EMA] => best val= {best_val:.4f}")
                return best_val
    return best_val

# -----------------------------------------
# Final Short Raw-Only Finetuning
# -----------------------------------------
def final_finetune_on_raw(model, ckpt_path, epochs=FINAL_FINE_EPOCHS, lr=FINAL_FINE_LR, batch_size=4):
    st = torch.load(ckpt_path, map_location=device)
    # Use strict=False here to ignore extra keys (e.g. from 'rgb_backbone')
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = True

    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv = LABELS_DIR / "ValidationLabels.csv"

    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    trset = VideoDatasetRawOnly(train_csv, FRAMES_DIR, 50, transform)
    valset = VideoDatasetRawOnly(val_csv, FRAMES_DIR, 50, transform)
    trld = DataLoader(trset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valld = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Define raw_forward for raw-only finetuning.
    def raw_forward(self, x):
        # x: (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        # Create or retrieve a dedicated RGB backbone for raw input.
        if not hasattr(self, 'rgb_backbone'):
            self.rgb_backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True, in_chans=3)
            self.rgb_backbone.reset_classifier(0)
            self.rgb_backbone.eval().to(device)
            for p in self.rgb_backbone.parameters():
                p.requires_grad = False
        with autocast(device_type='cuda', dtype=torch.float16):
            features = self.rgb_backbone(x)
        features = features.view(B, T, -1)
        # Create zeros for the flow branch (ignored)
        flow_zeros = torch.zeros_like(features, device=features.device)
        fused = torch.cat([features, flow_zeros], dim=2)  # (B,T,2*feat_dim_raw)
        fused = fused.permute(0,2,1)
        fused = self.cbam(fused)
        fused = fused.permute(0,2,1)
        lstm_out, (h_n, _) = self.bilstm(fused)
        temporal_ctx = torch.cat([h_n[-2], h_n[-1]], dim=1)
        spatial_feat = fused.mean(dim=1)
        sp = self.spatial_proj(spatial_feat)
        tp = self.temporal_proj(temporal_ctx)
        cross = self.cross_attn(sp, tp)
        cat = torch.cat([sp, cross], dim=1)
        cat = self.dropout(cat)
        logits = self.classifier(cat)
        return logits.view(B, 4, 4)

    model.raw_forward = raw_forward.__get__(model, model.__class__)

    criterion = WeightedOrdinalCrossEntropyLoss(LABEL_SMOOTHING).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_val = float('inf')
    es_count = 0
    early_stop = 4
    for e in range(epochs):
        model.train()
        run_loss = 0.0
        for frames, labs in tqdm(trld, desc=f"[RawFinetune E{e+1}/{epochs}]"):
            frames = frames.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outs = model.raw_forward(frames)
                loss = criterion(outs, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * frames.size(0)
        train_loss = run_loss / len(trld.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for frames, labs in valld:
                frames = frames.to(device)
                labs = labs.to(device)
                out = model.raw_forward(frames)
                ls = criterion(out, labs)
                val_loss += ls.item() * frames.size(0)
        val_loss /= len(valld.dataset)
        print(f"[RawFinetune] E{e+1} => train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            st2 = {
                "epoch": e+1,
                "val_loss": best_val,
                "model_state_dict": model.state_dict()
            }
            tmp = ckpt_path.with_suffix(".finetune_tmp")
            torch.save(st2, tmp)
            final_path = ckpt_path.with_suffix(".finetune")
            if final_path.exists():
                final_path.unlink()
            tmp.rename(final_path)
            es_count = 0
        else:
            es_count += 1
        if es_count >= early_stop:
            print(f"[RawFinetune] => early stop => best val= {best_val:.4f}")
            break

    final_path = ckpt_path.with_suffix(".finetune")
    if final_path.exists():
        st3 = torch.load(final_path, map_location=device)
        model.load_state_dict(st3["model_state_dict"], strict=False)
        print(f"[RawFinetune] => best val= {st3['val_loss']:.4f}")
    return model

# -----------------------------------------
# Evaluation Function
# -----------------------------------------
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, labs in tqdm(loader, desc="[Eval-dualLMDB]"):
            feats = feats.to(device)
            labs = labs.to(device)
            out = model(feats)
            preds = torch.argmax(out, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labs.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    for i, nm in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification => {nm}")
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{nm} Confusion")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


# -----------------------------------------
# Optuna Objective Function
# -----------------------------------------
def objective(trial):
    torch.cuda.empty_cache()
    gc.collect()
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [256, 512])
    lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
    drop_rate = trial.suggest_categorical("dropout_rate", [0.4, 0.5])
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.3, step=0.1)
    feat_dim = 2560  # assumed total feature dimension from two backbones
    total_ep = sum(e for _, e in PROG_SCHEDULE)
    model = DualStreamLMDBModel(
        feat_dim=feat_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout_rate=drop_rate,
        classifier_hidden=256
    ).to(device)
    trial_ckpt = MODEL_DIR / f"trial_{trial.number}_dual_ema_lr.pth"
    val_loss = progressive_train_dual_lmdb(
        model, total_ep, lr, trial_ckpt, batch_size,
        train_csv, val_csv, PROG_SCHEDULE,
        patience=EARLYSTOP_PATIENCE,
        gradient_accum_steps=GRADIENT_ACCUM_STEPS,
        weights_dict=weights_dict,
        mixup_alpha=mixup_alpha,
        ema_decay=0.995
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return val_loss

# -----------------------------------------
# Global CSV paths
# -----------------------------------------
train_csv = LABELS_DIR / "TrainLabels.csv"
val_csv = LABELS_DIR / "ValidationLabels.csv"
test_csv = LABELS_DIR / "TestLabels.csv"

# -----------------------------------------
# MAIN EXECUTION
# -----------------------------------------
def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    global weights_dict
    weights_dict = compute_class_weights_for_daisee(train_csv, val_csv)

    # Precompute & build LMDB for train/val/test for both rgb and flow
    for cfile in [train_csv, val_csv, test_csv]:
        for res in [112, 224, 300]:
            precompute_uniform_frames(cfile, FRAMES_DIR, NUM_FRAMES, res, flow=False)
            convert_to_lmdb(cfile, NUM_FRAMES, res, rgb=True, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
            precompute_uniform_frames(cfile, FRAMES_DIR, NUM_FRAMES, res, flow=True)
            convert_to_lmdb(cfile, NUM_FRAMES, res, rgb=False, transform=None, lmdb_map_size=LMDB_MAP_SIZE)

    db_path = BASE_DIR / "notebooks" / "dual_ema_lr.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = "dual_ema_lr_study"
    try:
        conn = sqlite3.connect(db_path)
        conn.close()
    except Exception as e:
        print(f"[DB error] => {e}")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=10),
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    target = 30
    while True:
        successes = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and 
                     t.value is not None and np.isfinite(t.value)]
        remain = target - len(successes)
        if remain <= 0:
            break
        print(f"[Optuna] => running {remain} more trial(s) to get {target}")
        study.optimize(objective, n_trials=remain, catch=(Exception,))
    best = min(successes, key=lambda t: t.value)
    print(f"[BestTrial] => {best.params}, val= {best.value:.4f}")

    # Final Training using best trial parameters (LMDB training)
    total_ep = sum(e for _, e in PROG_SCHEDULE)
    final_ckpt = MODEL_DIR / "final_dual_ema_lr_ckpt.pth"
    if not final_ckpt.exists():
        print("[Final-lmdbTrain+EMA+LR] => building best model from trial => no LR override")
        feat_dim = 2560
        final_model = DualStreamLMDBModel(
            feat_dim=feat_dim,
            lstm_hidden=best.params["lstm_hidden"],
            lstm_layers=best.params["lstm_layers"],
            dropout_rate=best.params["dropout_rate"],
            classifier_hidden=256
        ).to(device)
        val_loss = progressive_train_dual_lmdb(
            final_model, total_ep, best.params["lr"], final_ckpt,
            best.params["batch_size"],
            train_csv, val_csv, PROG_SCHEDULE,
            patience=EARLYSTOP_PATIENCE,
            gradient_accum_steps=GRADIENT_ACCUM_STEPS,
            weights_dict=weights_dict,
            mixup_alpha=best.params["mixup_alpha"],
            ema_decay=0.995
        )
        print(f"[Final-lmdbTrain+EMA+LR] => best val= {val_loss:.4f}")
    else:
        print("[Final-lmdbTrain+EMA+LR] => found => skip")

    # Final short raw pass (raw-only finetuning)
    final_model = DualStreamLMDBModel(
        feat_dim=2560,
        lstm_hidden=best.params["lstm_hidden"],
        lstm_layers=best.params["lstm_layers"],
        dropout_rate=best.params["dropout_rate"],
        classifier_hidden=256
    ).to(device)
    if not final_ckpt.with_suffix(".finetune").exists():
        print("[RawFinetune] => short pass => 300x300 => ignoring flow portion => LR=1e-5")
        final_finetune_on_raw(final_model, final_ckpt,
                              epochs=FINAL_FINE_EPOCHS,
                              lr=FINAL_FINE_LR,
                              batch_size=4)
    else:
        print("[RawFinetune] => found => skip")
        st = torch.load(final_ckpt.with_suffix(".finetune"), map_location=device)
        # Use strict=False here to ignore unexpected keys from the raw finetuning checkpoint
        final_model.load_state_dict(st["model_state_dict"], strict=False)
        final_model.to(device)

    # Evaluation on Test Set (using LMDB features from 300px)
    flow_test = convert_to_lmdb(test_csv, NUM_FRAMES, 300, rgb=False, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
    rgb_test = convert_to_lmdb(test_csv, NUM_FRAMES, 300, rgb=True, transform=None, lmdb_map_size=LMDB_MAP_SIZE)
    test_ds = HybridLMDBDataset(test_csv, rgb_test, flow_test, 300, NUM_FRAMES)
    test_ld = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    evaluate_model(final_model, test_ld)
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()
