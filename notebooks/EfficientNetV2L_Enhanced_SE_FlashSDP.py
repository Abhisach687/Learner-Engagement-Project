import os
import cv2
import gc
import torch
import optuna
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import pickle
import lmdb  # pip install lmdb
import logging
import sqlite3
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import timm  # pip install timm
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

# Enable PyTorch built-in Flash SDP (which uses FlashAttention when conditions are met)
torch.backends.cuda.enable_flash_sdp(True)

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONSTANTS & HYPERPARAMETERS
# ------------------------------
GRADIENT_ACCUM_STEPS = 8
NUM_FRAMES = 50
# Progressive schedule: first train at 224x224 then fine-tune at 300x300.
PROG_SCHEDULE = [(224, 15), (300, 10)]
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ------------------------------
# Environment & Paths
# ------------------------------
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

print("Models directory exists:", os.path.exists(MODEL_DIR))
print("Checkpoint path writable:", os.access(MODEL_DIR, os.W_OK))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Custom Collate Function
# ------------------------------
def custom_collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    return features, labels

# ------------------------------
# Image Transformations
# ------------------------------
def get_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ------------------------------
# Utility Functions
# ------------------------------
def get_csv_clip_id(video_stem: str) -> str:
    base = video_stem.strip()
    return base.replace("110001", "202614", 1) if base.startswith("110001") else base

def select_impactful_frames(video_folder: Path, num_frames=50):
    frame_files = sorted(video_folder.glob("frame_*.jpg"))
    total = len(frame_files)
    if total == 0:
        return []
    if total <= num_frames:
        return frame_files
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [frame_files[i] for i in indices]

# ------------------------------
# Precomputation & LMDB Caching Functions
# ------------------------------
def precompute_best_frames(csv_file: Path, video_root: Path, num_frames=50, resolution=224):
    data = pd.read_csv(csv_file, dtype=str)
    data.columns = data.columns.str.strip()
    split = csv_file.stem.replace("Labels", "").strip()
    valid_indices = []
    precomputed = []
    skipped = 0
    for idx, row in tqdm(data.iterrows(), total=len(data),
                         desc=f"Precomputing frames for {csv_file.stem} at {resolution}x{resolution}"):
        clip_id = get_csv_clip_id(row["ClipID"].split('.')[0])
        video_folder = video_root / split / clip_id
        if video_folder.exists():
            frames = select_impactful_frames(video_folder, num_frames)
            if len(frames) >= num_frames:
                precomputed.append(frames[:num_frames])
                valid_indices.append(idx)
            else:
                skipped += 1
        else:
            skipped += 1
    print(f"Precomputation: Skipped {skipped} videos out of {len(data)}.")
    cache_data = {"valid_indices": valid_indices, "precomputed_frames": precomputed}
    cache_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Precomputed results saved to {cache_file}")
    return cache_data

def convert_pkl_to_lmdb(csv_file: Path, num_frames=50, resolution=224,
                          transform=None, lmdb_map_size=1 * 1024**3):
    if transform is None:
        transform = get_transform(resolution)
    pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
    lmdb_path = CACHE_DIR / f"lmdb_{csv_file.stem}_frame_{num_frames}_{resolution}"
    if (lmdb_path / "data.mdb").exists():
        print(f"LMDB database already exists at {lmdb_path}")
        return lmdb_path
    env = lmdb.open(str(lmdb_path), map_size=lmdb_map_size)
    if not pkl_file.exists():
        precompute_best_frames(csv_file, FRAMES_DIR, num_frames=num_frames, resolution=resolution)
    with open(pkl_file, "rb") as f:
        cache = pickle.load(f)
    valid_indices = cache["valid_indices"]
    file_paths_list = cache["precomputed_frames"]
    backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
    backbone.reset_classifier(0)
    backbone.eval()
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Converting frame paths to LMDB features for {csv_file.stem} at {resolution}x{resolution} ...")
    with env.begin(write=True) as txn:
        for idx, paths in tqdm(enumerate(file_paths_list), total=len(file_paths_list)):
            video_features = []
            for fp in paths:
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    img = Image.new('RGB', (resolution, resolution))
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    feat = backbone(tensor)
                    feat = feat.squeeze(0).cpu().half().detach()
                video_features.append(feat.numpy())
            if video_features:
                video_features_np = np.stack(video_features)
                key = f"video_{valid_indices[idx]}".encode("utf-8")
                txn.put(key, pickle.dumps(video_features_np))
    env.close()
    print(f"LMDB database created at {lmdb_path}")
    return lmdb_path

# ------------------------------
# LMDB Dataset Classes
# ------------------------------
class VideoDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self, csv_file, lmdb_path, num_frames=50, resolution=224):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        self.resolution = resolution
        pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
        with open(pkl_file, "rb") as f:
            cache = pickle.load(f)
        self.valid_indices = cache["valid_indices"]
        self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)
        self.num_frames = num_frames
        self.lmdb_path = str(lmdb_path)
        self.env = None
    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        env = self._init_env()
        original_idx = self.valid_indices[idx]
        key = f"video_{original_idx}".encode("utf-8")
        with env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise IndexError(f"Key {key} not found in LMDB")
            features_np = pickle.loads(data_bytes)
            features = torch.from_numpy(features_np)
        labels_array = np.array(self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].tolist(), dtype=np.int64)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)
        return features, labels_tensor

class VideoDatasetRaw(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_root, num_frames=50, transform=None, split="train"):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        self.split = split
        pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_raw.pkl"
        if not pkl_file.exists():
            cache = precompute_best_frames(csv_file, video_root, num_frames=num_frames)
            with open(pkl_file, "wb") as f:
                pickle.dump(cache, f)
        else:
            with open(pkl_file, "rb") as f:
                cache = pickle.load(f)
        self.valid_indices = cache["valid_indices"]
        self.file_paths = cache["precomputed_frames"]
        self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)
        self.num_frames = num_frames
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        paths = self.file_paths[idx]
        frames = []
        for fp in paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new('RGB', (self.transform.transforms[0].size, self.transform.transforms[0].size))
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video_tensor = torch.stack(frames)
        labels_array = np.array(self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].tolist(), dtype=np.int64)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)
        return video_tensor, labels_tensor

# ------------------------------
# Class-Specific Focal Loss
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, target):
        num_classes = logits.size(1)
        target_onehot = F.one_hot(target, num_classes=num_classes).float()
        probs = F.softmax(logits, dim=1)
        pt = (probs * target_onehot).sum(dim=1)
        logpt = torch.log(pt + 1e-8)
        if self.alpha is not None:
            alpha_t = (self.alpha.to(logits.device) * target_onehot).sum(dim=1)
        else:
            alpha_t = 1.0
        if self.gamma.numel() > 1:
            gamma_t = (self.gamma.to(logits.device) * target_onehot).sum(dim=1)
        else:
            gamma_t = self.gamma.to(logits.device)
        loss = -alpha_t * ((1 - pt) ** gamma_t) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Create per-head focal loss functions
loss_engagement  = FocalLoss(alpha=[1.0, 1.0, 1.0, 1.0], gamma=2.5)
loss_boredom     = FocalLoss(alpha=[1.0, 1.0, 1.0, 1.0], gamma=2.0)
loss_confusion   = FocalLoss(alpha=[1.0, 1.0, 1.0, 1.0], gamma=1.5)
loss_frustration = FocalLoss(alpha=[1.0, 1.0, 1.0, 1.0], gamma=2.0)

# ------------------------------
# Temporal CutMix Data Augmentation
# ------------------------------
class TemporalCutMix(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, video_batch, labels=None):
        if torch.rand(1).item() > self.p:
            return video_batch, labels
        B, T, C, H, W = video_batch.shape
        rand_indices = torch.randperm(B, device=video_batch.device)
        t_start = torch.randint(0, T-10, (B,), device=video_batch.device)
        t_end = t_start + 10
        for i in range(B):
            video_batch[i, t_start[i]:t_end[i]] = video_batch[rand_indices[i], t_start[i]:t_end[i]]
            if labels is not None:
                labels[i, t_start[i]:t_end[i]] = labels[rand_indices[i], t_start[i]:t_end[i]]
        return video_batch, labels

# ------------------------------
# Spatial SE Module (Replacing CBAM)
# ------------------------------
class SpatialSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x).transpose(1, 2)
        y = self.fc(y).transpose(1, 2)
        return x * y.expand_as(x)

# ------------------------------
# Temporal Module using PyTorch’s Built-In Flash Attention
# ------------------------------
class TemporalFlashSDPModule(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(TemporalFlashSDPModule, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.0)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_size)
        attn_out = F.scaled_dot_product_attention(lstm_out, lstm_out, lstm_out, dropout_p=0.0, is_causal=False)
        out = self.layer_norm(lstm_out + attn_out)
        return out

# ------------------------------
# Enhanced Model Architecture using PyTorch Built-In Flash Attention
# ------------------------------
class EfficientNetV2L_Enhanced(nn.Module):
    def __init__(self, lstm_hidden=512, dropout_rate=0.5, n_heads=4):
        super(EfficientNetV2L_Enhanced, self).__init__()
        # Backbone: EfficientNetV2-L with minimal freezing
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < 6:
                    for param in block.parameters():
                        param.requires_grad = False
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = 1536
        # Use SE for spatial recalibration
        self.se = SpatialSELayer(channel=self.feature_dim, reduction=16)
        # Temporal module: BiLSTM + Flash-based attention
        self.temporal_module = TemporalFlashSDPModule(input_dim=self.feature_dim, hidden_size=lstm_hidden)
        self.d_model = 2 * lstm_hidden
        # Project global spatial feature to d_model
        self.spatial_proj = nn.Linear(self.feature_dim, self.d_model)
        # Multi-head attention pooling
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, batch_first=True)
        self.ln_fusion = nn.LayerNorm(self.d_model * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.d_model * 2, 16)  # 16 logits, to be reshaped into (4,4)
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)
            features = self.backbone(x)
            features = features.view(B, T, self.feature_dim)
        elif x.dim() == 3:
            features = x
            B, T, _ = features.size()
        else:
            raise ValueError("Input tensor must have 3 or 5 dimensions.")
        # Apply SE: permute to (B, C, T)
        features_se = self.se(features.permute(0, 2, 1)).permute(0, 2, 1)
        global_spatial = features_se.mean(dim=1)  # (B, feature_dim)
        global_spatial_proj = self.spatial_proj(global_spatial)  # (B, d_model)
        temporal_features = self.temporal_module(features_se)  # (B, T, d_model)
        # Use global spatial as query for attention pooling
        query = global_spatial_proj.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.cross_attn(query, temporal_features, temporal_features)
        attn_out = attn_out.squeeze(1)  # (B, d_model)
        fusion = torch.cat((global_spatial_proj, attn_out), dim=1)  # (B, 2*d_model)
        fusion = self.ln_fusion(fusion)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # (B, 16)
        # Reshape logits to (B, 4, 4) and return as 4 separate outputs
        B = logits.size(0)
        logits = logits.view(B, 4, 4)
        # Return separate outputs for each affective state
        return logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3]

# ------------------------------
# Training Function (Progressive Resolution, Mixed Precision, Grad Accumulation)
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                            patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": other_params, "lr": lr}
    ], weight_decay=1e-4)
    scaler = GradScaler()
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_epoch = 0
    for res, ep in PROG_SCHEDULE:
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res, transform=transform)
        val_lmdb = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res, transform=transform)
        train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set = VideoDatasetLMDB(val_csv, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        for epoch in range(ep):
            print(f"Progressive Training: Epoch {current_epoch+1}/{total_epochs} at resolution {res}x{res}")
            model.train()
            running_loss = 0.0
            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                with autocast(device_type='cuda', dtype=torch.float16):
                    out_eng, out_bor, out_conf, out_frust = model(features)
                    loss = (loss_engagement(out_eng, labels[:, 0]) +
                            loss_boredom(out_bor, labels[:, 1]) +
                            loss_confusion(out_conf, labels[:, 2]) +
                            loss_frustration(out_frust, labels[:, 3])) / 4.0
                scaler.scale(loss / gradient_accum_steps).backward()
                if (i + 1) % gradient_accum_steps == 0:
                    scaler.step(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item() * features.size(0)
                del features, labels, out_eng, out_bor, out_conf, out_frust, loss
                if (i + 1) % 30 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            train_loss = running_loss / len(train_loader.dataset)
            model.eval()
            val_loss = 0.0
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                for features, labels in val_loader:
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                    out_eng, out_bor, out_conf, out_frust = model(features)
                    loss = (loss_engagement(out_eng, labels[:, 0]) +
                            loss_boredom(out_bor, labels[:, 1]) +
                            loss_confusion(out_conf, labels[:, 2]) +
                            loss_frustration(out_frust, labels[:, 3])) / 4.0
                    val_loss += loss.item() * features.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {current_epoch+1}/{total_epochs} at {res}x{res} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state = {
                    "epoch": current_epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                temp_path = checkpoint_path.with_suffix(".tmp")
                torch.save(state, temp_path, _use_new_zipfile_serialization=False)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                temp_path.rename(checkpoint_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {current_epoch+1}. Best val loss: {best_val_loss:.4f}")
                return best_val_loss
            current_epoch += 1
    return best_val_loss

# ------------------------------
# Evaluation Function
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = {0:[], 1:[], 2:[], 3:[]}
    all_labels = {0:[], 1:[], 2:[], 3:[]}
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            out_eng, out_bor, out_conf, out_frust = model(frames)
            pred_eng = torch.argmax(out_eng, dim=1)
            pred_bor = torch.argmax(out_bor, dim=1)
            pred_conf = torch.argmax(out_conf, dim=1)
            pred_frust = torch.argmax(out_frust, dim=1)
            all_preds[0].append(pred_eng.cpu())
            all_preds[1].append(pred_bor.cpu())
            all_preds[2].append(pred_conf.cpu())
            all_preds[3].append(pred_frust.cpu())
            all_labels[0].append(labels[:, 0].cpu())
            all_labels[1].append(labels[:, 1].cpu())
            all_labels[2].append(labels[:, 2].cpu())
            all_labels[3].append(labels[:, 3].cpu())
    for k in all_preds.keys():
        all_preds[k] = torch.cat(all_preds[k]).numpy()
        all_labels[k] = torch.cat(all_labels[k]).numpy()
    for i, state in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification report for {state}:")
        print(classification_report(all_labels[i], all_preds[i], digits=3))
        cm = confusion_matrix(all_labels[i], all_preds[i])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {state}")
        plt.colorbar()
        plt.xticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
        plt.yticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # CSV file paths
    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv = LABELS_DIR / "ValidationLabels.csv"
    test_csv = LABELS_DIR / "TestLabels.csv"
    
    # Precompute caches and LMDB databases for resolutions 224 and 300
    resolutions = [224, 300]
    for csv in [train_csv, val_csv, test_csv]:
        for res in resolutions:
            precompute_best_frames(csv, FRAMES_DIR, num_frames=NUM_FRAMES, resolution=res)
            convert_pkl_to_lmdb(csv, num_frames=NUM_FRAMES, resolution=res,
                                transform=get_transform(res), lmdb_map_size=1 * 1024**3)
    
    # ------------------------------
    # Hyperparameter Tuning using Progressive Training
    # ------------------------------
    def objective(trial):
        torch.cuda.empty_cache()
        gc.collect()
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [512, 768])
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.5, 0.6])
        total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
        model = EfficientNetV2L_Enhanced(lstm_hidden=lstm_hidden, n_heads=n_heads, dropout_rate=dropout_rate).to(device)
        trial_checkpoint = MODEL_DIR / f"trial_eff_v2l_{trial.number}_enhanced_flashSDP_checkpoint.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        loss = progressive_train_model(model, total_epochs, lr, trial_checkpoint, batch_size,
                                       patience=3, gradient_accum_steps=GRADIENT_ACCUM_STEPS)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return loss

    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_enhanced_flashSDP.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        print(f"Database created/connected at: {db_path}")
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=10),
        study_name="efficientnetv2l_enhanced_study",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    target_trials = 30
    while True:
        successes = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
        remaining = target_trials - len(successes)
        if remaining <= 0:
            break
        print(f"Running {remaining} additional trial(s) to reach {target_trials} successful trials...")
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    print(f"Optuna tuning complete. Total successful trials: {len(successes)}")
    best_trial = min(successes, key=lambda t: t.value)
    print(f"Best trial parameters: {best_trial.params}")
    
    # ------------------------------
    # Final Training (using raw images for end-to-end fine-tuning)
    # ------------------------------
    total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l_enhanced_flashSDP_checkpoint.pth"
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        params = best_trial.params
        batch_size = params.get("batch_size", 4)
        lr = params.get("lr", 1e-4)
        lstm_hidden = params.get("lstm_hidden", 512)
        n_heads = params.get("n_heads", 4)
        dropout_rate = params.get("dropout_rate", 0.5)
        final_model = EfficientNetV2L_Enhanced(lstm_hidden=lstm_hidden, n_heads=n_heads, dropout_rate=dropout_rate).to(device)
        # Unfreeze additional backbone layers for fine-tuning
        if hasattr(final_model.backbone, "blocks"):
            for i, block in enumerate(final_model.backbone.blocks):
                if i >= 6:
                    for param in block.parameters():
                        param.requires_grad = True
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_loss = progressive_train_model(final_model, total_epochs, lr, final_checkpoint, batch_size,
                                             patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS)
    else:
        print("\n--- Skipping Final Training (Checkpoint Exists) ---")
        print(f"Using existing model from: {final_checkpoint}")
    
    # ------------------------------
    # Evaluation on Test Set (300x300)
    # ------------------------------
    test_transform = get_transform(300)
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
    eval_model = EfficientNetV2L_Enhanced(lstm_hidden=best_trial.params.get("lstm_hidden", 512),
                                           n_heads=best_trial.params.get("n_heads", 4),
                                           dropout_rate=best_trial.params.get("dropout_rate", 0.5)).to(device)
    state = torch.load(final_checkpoint, map_location=device)
    eval_model.load_state_dict(state["model_state_dict"])
    eval_model.to(device)
    evaluate_model(eval_model, test_loader)
    torch.cuda.empty_cache()
    gc.collect()
    print("\n--- Evaluation Complete ---")
