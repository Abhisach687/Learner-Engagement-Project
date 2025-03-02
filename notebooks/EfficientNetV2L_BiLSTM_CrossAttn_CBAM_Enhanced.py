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

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONSTANTS & HYPERPARAMETERS
# ------------------------------
GRADIENT_ACCUM_STEPS = 4        # Accumulate gradients over mini-batches
NUM_FRAMES = 50
# Progressive resolution schedule: (resolution, epochs)
PROG_SCHEDULE = [(112, 5), (224, 10), (300, 15)]
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
# Image Transformations
# ------------------------------
def get_transform(resolution):
    # Here you could add additional augmentations (e.g. RandAugment, MixUp) at higher resolutions.
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
# Precomputation & Caching Functions
# (Assume LMDB files for 112, 224, and 300 already exist)
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
        labels = self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return features, torch.tensor(labels.values, dtype=torch.long)

class VideoDatasetRaw(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_root, num_frames=50, transform=None):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
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
        labels = self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return video_tensor, torch.tensor(labels.values, dtype=torch.long)

# ------------------------------
# Focal Loss Implementation
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ------------------------------
# Temporal Module: Stacked BiLSTM + Multi-Head Self-Attention
# ------------------------------
class TemporalModule(nn.Module):
    def __init__(self, input_dim, lstm_hidden, n_layers=2, n_heads=4):
        super(TemporalModule, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        # Output dimension = 2 * lstm_hidden
        self.self_attn1 = nn.MultiheadAttention(embed_dim=2*lstm_hidden, num_heads=n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(2*lstm_hidden)
        self.self_attn2 = nn.MultiheadAttention(embed_dim=2*lstm_hidden, num_heads=n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(2*lstm_hidden)
    
    def forward(self, x):
        # x: (B, T, input_dim)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)
        attn_out1, _ = self.self_attn1(lstm_out, lstm_out, lstm_out)
        out1 = self.ln1(lstm_out + attn_out1)
        attn_out2, _ = self.self_attn2(out1, out1, out1)
        out2 = self.ln2(out1 + attn_out2)
        return out2  # (B, T, 2*lstm_hidden)
    
    # CBAM Module for Temporal Attention 
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, T)
        channel_attn = self.channel_attention(x)
        x = x * channel_attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_attn
        return x


# ------------------------------
# Multi-Head Cross-Attention Block for Fusion
# ------------------------------
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super(MultiHeadCrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    
    def forward(self, query, key, value):
        # query: (B, 1, d_model), key & value: (B, T, d_model)
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output  # (B, 1, d_model)

# ------------------------------
# Enhanced Model Architecture
# ------------------------------
class EfficientNetV2L_Enhanced(nn.Module):
    def __init__(self, lstm_hidden=512, n_heads=4, dropout_rate=0.5):
        super(EfficientNetV2L_Enhanced, self).__init__()
        # Backbone: EfficientNetV2-L with minimal freezing (freeze early layers: stages 0-5)
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
        
        # Spatial refinement using CBAM
        self.cbam = CBAM(self.feature_dim, reduction=16, kernel_size=7)
        
        # Temporal module: stacked two-layer bidirectional LSTM + multi-head self-attention
        self.temporal_module = TemporalModule(input_dim=self.feature_dim, lstm_hidden=lstm_hidden,
                                               n_layers=2, n_heads=n_heads)
        self.d_model = 2 * lstm_hidden  # output dimension from temporal module
        
        # Project global spatial feature (obtained by temporal averaging of CBAM-enhanced features) to d_model
        self.spatial_proj = nn.Linear(self.feature_dim, self.d_model)
        
        # Multi-head cross-attention block to fuse spatial and temporal context
        self.cross_attn = MultiHeadCrossAttention(d_model=self.d_model, n_heads=n_heads)
        
        self.ln_fusion = nn.LayerNorm(self.d_model * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.d_model * 2, 16)
    
    def forward(self, x):
        # Support raw images (B, T, C, H, W) or precomputed features (B, T, feature_dim)
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)
            features = self.backbone(x)  # (B*T, feature_dim)
            features = features.view(B, T, self.feature_dim)
        elif x.dim() == 3:
            features = x
            B, T, _ = features.size()
        else:
            raise ValueError("Input tensor must have 3 or 5 dimensions.")
        
        # Apply CBAM
        features_cbam = features.permute(0, 2, 1)  # (B, feature_dim, T)
        features_cbam = self.cbam(features_cbam)
        features_cbam = features_cbam.permute(0, 2, 1)  # (B, T, feature_dim)
        
        # Global spatial feature: temporal average then project to d_model
        global_spatial = features_cbam.mean(dim=1)  # (B, feature_dim)
        global_spatial_proj = self.spatial_proj(global_spatial)  # (B, d_model)
        
        # Temporal module: refine temporal context
        temporal_features = self.temporal_module(features_cbam)  # (B, T, d_model)
        
        # Use global spatial projection as query (unsqueeze to shape (B, 1, d_model))
        query = global_spatial_proj.unsqueeze(1)
        cross_attn_out = self.cross_attn(query, temporal_features, temporal_features)  # (B, 1, d_model)
        cross_attn_out = cross_attn_out.squeeze(1)  # (B, d_model)
        
        # Fusion: concatenate the spatial and cross-attended temporal context
        fusion = torch.cat((global_spatial_proj, cross_attn_out), dim=1)  # (B, 2*d_model)
        fusion = self.ln_fusion(fusion)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # (B, 16)
        return logits.view(B, 4, 4)

# ------------------------------
# Training Function with Progressive Resolution, Mixed Precision & Gradient Accumulation
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                            patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
    # Use parameter ID comparison to separate backbone parameters from new layers.
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": other_params, "lr": lr}
    ], weight_decay=1e-4)
    scaler = GradScaler()
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_epoch = 0
    for res, ep in PROG_SCHEDULE:
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res,
                                          transform=transform, lmdb_map_size=1 * 1024**3)
        val_lmdb = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res,
                                        transform=transform, lmdb_map_size=1 * 1024**3)
        train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set = VideoDatasetLMDB(val_csv, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for epoch in range(ep):
            print(f"Progressive Training: Epoch {current_epoch+1}/{total_epochs} at resolution {res}x{res}")
            model.train()
            running_loss = 0.0
            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(features)
                    outputs = outputs.view(outputs.size(0), 4, 4)
                    loss = sum(focal_loss(outputs[:, d], labels[:, d]) for d in range(4)) / 4.0
                scaler.scale(loss / gradient_accum_steps).backward()
                if (i + 1) % gradient_accum_steps == 0:
                    scaler.step(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item() * features.size(0)
                del features, labels, outputs, loss
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
                    outputs = model(features)
                    outputs = outputs.view(outputs.size(0), 4, 4)
                    loss = sum(focal_loss(outputs[:, d], labels[:, d]) for d in range(4)) / 4.0
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
    all_preds = []
    all_labels = []
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(frames)
            outputs = outputs.view(outputs.size(0), 4, 4)
            preds = torch.argmax(outputs, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    for i, metric in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification report for {metric}:")
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {metric}")
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
    # Precompute caches for all resolutions
    resolutions = [112, 224, 300]
    for csv in [train_csv, val_csv, test_csv]:
        for res in resolutions:
            precompute_best_frames(csv, FRAMES_DIR, num_frames=NUM_FRAMES, resolution=res)
            convert_pkl_to_lmdb(csv, num_frames=NUM_FRAMES, resolution=res,
                                transform=get_transform(res), lmdb_map_size=1 * 1024**3)
    # ------------------------------
    # Hyperparameter Tuning using Progressive Training over All 3 Resolutions
    # ------------------------------
    def objective(trial):
        torch.cuda.empty_cache()
        gc.collect()
        batch_size = 8  
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [512, 768])
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.5, 0.6])
        total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
        model = EfficientNetV2L_Enhanced(lstm_hidden=lstm_hidden, n_heads=n_heads, dropout_rate=dropout_rate).to(device)
        trial_checkpoint = MODEL_DIR / f"trial_eff_v2l_{trial.number}_enhanced_checkpoint.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        loss = progressive_train_model(model, total_epochs, lr, trial_checkpoint, batch_size,
                                       patience=3, gradient_accum_steps=GRADIENT_ACCUM_STEPS)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return loss

    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_bilstm_crossattn_cbam_enhanced.db"
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
        study_name="efficientnetv2l_bilstm_crossattn_cbam_study",
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
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l_enhanced_checkpoint.pth"
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        params = best_trial.params
        batch_size = 8
        lr = params.get("lr", 1e-4)
        lstm_hidden = params.get("lstm_hidden", 512)
        n_heads = params.get("n_heads", 4)
        dropout_rate = params.get("dropout_rate", 0.5)
        final_model = EfficientNetV2L_Enhanced(lstm_hidden=lstm_hidden, n_heads=n_heads, dropout_rate=dropout_rate).to(device)
        # Unfreeze backbone layers after stage 5 for fine-tuning
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
    # Evaluation on Test Set using Highest Resolution (300x300)
    # ------------------------------
    test_transform = get_transform(300)
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
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
