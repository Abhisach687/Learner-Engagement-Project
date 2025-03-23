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
import io
import lmdb  # pip install lmdb
import logging
import sqlite3
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import timm  # pip install timm
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONSTANTS & HYPERPARAMETERS
# ------------------------------
NUM_FRAMES = 50
# Progressive resolution schedule: (resolution, epochs)
PROG_SCHEDULE = [(112, 6), (224, 10), (300, 6)]
BATCH_SIZES = {
    112: 8,  # Smaller images, larger batch
    224: 6,  # Medium resolution
    300: 4   # High resolution, smaller batch
}
GRAD_ACCUM = {
    112: 2,  # Less accumulation at lower res
    224: 4,  # More accumulation as resolution increases
    300: 6   # Higher accumulation for largest images
}
LEARNING_RATES = {
    112: 1.5e-5,  # Start with slightly higher LR
    224: 7e-6,    # Lower as we progress
    300: 3e-6     # Lowest for fine-tuning
}
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
    # If LMDB already exists, return it.
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

    # Use EfficientNetV2-L (tf variant) from timm with frozen backbone
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
                    feat = feat.squeeze(0).cpu().half().detach()  # ensure detached and on CPU
                # Convert to NumPy array to avoid pickling issues with torch tensors
                video_features.append(feat.numpy())
            if video_features:
                video_features_np = np.stack(video_features)  # shape: (num_frames, feature_dim)
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
        video_tensor = torch.stack(frames)  # (T, C, H, W)
        labels = self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return video_tensor, torch.tensor(labels.values, dtype=torch.long)

#------------------------------
# Class-Balanced Focal Loss Implementation
# ------------------------------
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, class_weights=None):
        """
        inputs: (B, 4) logits for one task
        targets: (B,) class indices for one task
        class_weights: (4,) weight for each class in this task
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Focal loss scaling
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if class_weights is not None:
            weight = class_weights[targets]
            loss = weight * focal_weight * ce_loss
        else:
            # Use default alpha from original settings if no weights
            alpha = FOCAL_ALPHA
            loss = alpha * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        

# ------------------------------
# Class Weight Computation
# ------------------------------
def compute_class_weights(csv_file):
    data = pd.read_csv(csv_file, dtype=str)
    data.columns = data.columns.str.strip()
    weights = []
    
    for col in ["Engagement", "Boredom", "Confusion", "Frustration"]:
        # Count instances per class
        counts = data[col].astype(int).value_counts().sort_index()
        # Inverse frequency weighting with smoothing
        w = 1.0 / (counts + 10)  # Adding 10 prevents extreme weights
        # Normalize weights to average of 1.0
        w = w / w.mean()
        weights.append(torch.tensor(w.values, dtype=torch.float32))
    
    return weights

# ------------------------------
# CBAM & Cross-Attention Modules
# ------------------------------
class BalancedCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Lightweight channel attention - critical for preventing class collapse
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Reduced bottleneck size to minimize parameters
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channels, max(channels // reduction, 32), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(channels // reduction, 32), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (B, C, T)
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out  # Only channel attention, no spatial attention

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, spatial_feat, temporal_feat):
        # spatial_feat: (B, feature_dim)
        # temporal_feat: (B, feature_dim)
        query = self.query(spatial_feat).unsqueeze(1)  # (B, 1, feature_dim)
        key = self.key(temporal_feat).unsqueeze(2)       # (B, feature_dim, 1)
        attn = self.softmax(torch.bmm(query, key))       # (B, 1, 1)
        value = self.value(temporal_feat)
        return attn.squeeze(-1) * value  # (B, feature_dim)
    
# ------------------------------
# Enhanced BiLSTM Module
# ------------------------------
    
class EnhancedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x):
        # x: (B, T, input_dim)
        outputs, (h_n, _) = self.bilstm(x)
        
        # Get final states from both directions
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*hidden_dim)
        
        # Apply layer normalization (improves stability)
        final_hidden = self.layernorm(final_hidden)
        
        # Add residual connection with mean-pooled outputs (better gradient flow)
        mean_pooled = torch.mean(outputs, dim=1)  # (B, 2*hidden_dim)
        enhanced = final_hidden + 0.2 * mean_pooled
        
        return outputs, enhanced

# ------------------------------
# Model Architecture
# ------------------------------
class EfficientNetV2L_EnhancedBiLSTM(nn.Module):
    def __init__(self, lstm_hidden=384, dropout_rate=0.4):
        super().__init__()
        # EfficientNetV2L backbone (unchanged - proven effective)
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        
        # Get feature dimension from backbone
        self.feature_dim = getattr(self.backbone, "num_features", 1536)
        
        # Enhanced BiLSTM with residual connections
        self.bilstm = EnhancedBiLSTM(
            input_dim=self.feature_dim, 
            hidden_dim=lstm_hidden,
            num_layers=2, 
            dropout=0.3
        )
        
        # Balanced CBAM (channel-only attention)
        self.cbam = BalancedCBAM(self.feature_dim, reduction=16)
        
        # Cross-attention mechanism (from your best model)
        self.spatial_proj = nn.Linear(self.feature_dim, 256)
        self.temporal_proj = nn.Linear(lstm_hidden * 2, 256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization for stable training
        self.fusion_bn = nn.BatchNorm1d(512)
        
        # Final classifier (4 tasks with 4 classes each)
        self.classifier = nn.Linear(512, 16)
    
    def forward(self, x):
        B = x.size(0)
        
        # Process raw frames or precomputed features
        if x.dim() == 5:  # Raw frames: (B, T, C, H, W)
            T = x.size(1)
            x = x.view(B*T, x.size(2), x.size(3), x.size(4))
            with autocast(enabled=False, device_type='cuda'):  # Use fp32 for backbone to prevent instability
                features = self.backbone(x)  # (B*T, feature_dim)
            features = features.view(B, T, -1)  # (B, T, feature_dim)
        else:  # Precomputed features: (B, T, feature_dim)
            features = x
            T = x.size(1)
        
        # Apply channel-only CBAM (prevents class collapse)
        features_t = features.permute(0, 2, 1)  # (B, feature_dim, T)
        features_t = self.cbam(features_t)
        features = features_t.permute(0, 2, 1)  # (B, T, feature_dim)
        
        # Process with BiLSTM
        lstm_outputs, temporal_feat = self.bilstm(features)
        
        # Get spatial features
        spatial_feat = torch.mean(features, dim=1)  # Average over time
        
        # Project features to common space for cross-attention
        spatial_proj = self.spatial_proj(spatial_feat)  # (B, 256)
        temporal_proj = self.temporal_proj(temporal_feat)  # (B, 256)
        
        # Combine features
        fusion = torch.cat((spatial_proj, temporal_proj), dim=1)  # (B, 512)
        fusion = self.fusion_bn(fusion)
        fusion = self.dropout(fusion)
        
        # Generate outputs for all tasks
        logits = self.classifier(fusion)  # (B, 16)
        
        # Reshape to (B, 4, 4) for 4 tasks with 4 classes each
        return logits.view(B, 4, 4)
    
# ------------------------------
# Training Function with Progressive Resolution, Mixed Precision & Gradient Accumulation
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                            patience=5):
    # Compute class weights once at start
    train_weights = compute_class_weights(train_csv)
    
    # Track best model
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    current_epoch = 0
    for res_idx, (res, epochs) in enumerate(PROG_SCHEDULE):
        print(f"\n=== Training at {res}x{res} resolution ===")
        
        # Configure for current resolution
        batch_size = BATCH_SIZES[res]
        grad_steps = GRAD_ACCUM[res]
        base_lr = LEARNING_RATES[res]
        
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res,
                                          transform=transform, lmdb_map_size=1 * 1024**3)
        val_lmdb = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res,
                                        transform=transform, lmdb_map_size=1 * 1024**3)
        train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set = VideoDatasetLMDB(val_csv, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Progressive unfreezing strategy
        if res_idx == 0:  # First stage: freeze most backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            # Only unfreeze last 2 blocks
            if hasattr(model.backbone, "blocks"):
                for i in range(len(model.backbone.blocks) - 2, len(model.backbone.blocks)):
                    for param in model.backbone.blocks[i].parameters():
                        param.requires_grad = True
        
        elif res_idx == 1:  # Second stage: unfreeze more layers
            if hasattr(model.backbone, "blocks"):
                for i in range(len(model.backbone.blocks) - 5, len(model.backbone.blocks)):
                    for param in model.backbone.blocks[i].parameters():
                        param.requires_grad = True
        
        else:  # Final stage: unfreeze everything
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        # Create optimizer with parameter groups
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        other_params = [p for n, p in model.named_parameters() 
                       if "backbone" not in n and p.requires_grad]
        
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": base_lr * 0.1},  # Lower LR for backbone
            {"params": other_params, "lr": base_lr}
        ], weight_decay=1e-4)
        
        # LR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=base_lr * 0.1
        )
        
        loss_fn = ClassBalancedFocalLoss(gamma=FOCAL_GAMMA)
        
        for epoch in range(epochs):
            print(f"Progressive Training: Epoch {current_epoch+1}/{total_epochs} at resolution {res}x{res}")
            model.train()
            running_loss = 0.0
            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(features)
                    outputs = outputs.view(outputs.size(0), 4, 4)
                    
                    # Compute loss for each task with class weights
                    task_losses = [
                        loss_fn(outputs[:, d], labels[:, d], train_weights[d].to(device))
                        for d in range(4)
                    ]
                    loss = sum(task_losses) / 4.0
                
                # Scale loss and backprop
                scaler.scale(loss / grad_steps).backward()
                
                if (i + 1) % grad_steps == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                running_loss += loss.item() * features.size(0)
                del features, labels, outputs, loss
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            train_loss = running_loss / len(train_loader.dataset)
            
            # Update LR
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                for features, labels in val_loader:
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(features)
                    outputs = outputs.view(outputs.size(0), 4, 4)
                    
                    # Compute validation loss with class weights
                    task_losses = [
                        loss_fn(outputs[:, d], labels[:, d], train_weights[d].to(device))
                        for d in range(4)
                    ]
                    loss = sum(task_losses) / 4.0
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
                
                # Save resolution-specific checkpoint
                res_checkpoint = checkpoint_path.parent / f"best_model_balanced_res{res}.pth"
                torch.save(state, res_checkpoint)
                
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {current_epoch+1}. Best val loss: {best_val_loss:.4f}")
                
                # Load best weights for this resolution before moving to next resolution
                res_checkpoint = checkpoint_path.parent / f"best_model_balanced_res{res}.pth"
                if res_checkpoint.exists():
                    checkpoint = torch.load(res_checkpoint, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                
                break
            
            current_epoch += 1
            
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
    
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
            
            # Original prediction
            outputs = model(frames)
            
            # Simple horizontal flip TTA
            flipped = torch.flip(frames, dims=[-1])
            flip_outputs = model(flipped)
            
            # Average predictions
            outputs = (outputs + flip_outputs) / 2.0
            
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
    
    # Precompute caches and LMDB for each resolution
    resolutions = [res for res, _ in PROG_SCHEDULE]
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
        batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])
        lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [256, 384])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.3, 0.4, 0.5])
        total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
        model = EfficientNetV2L_EnhancedBiLSTM(lstm_hidden=lstm_hidden, 
                                               dropout_rate=dropout_rate).to(device)
        trial_checkpoint = MODEL_DIR / f"trial_eff_v2l_enhanced_balanced_{trial.number}_checkpoint.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        loss = progressive_train_model(model, total_epochs, lr, trial_checkpoint, batch_size,
                                       patience=3)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return loss
    
    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_enhanced_balanced_bilstm.db"
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
        study_name="efficientnetv2l_enhanced_bilstm_study",
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
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l_enhanced_bilstm_balanced__checkpoint.pth"
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        params = best_trial.params
        batch_size = params.get("batch_size", 6)
        lr = params.get("lr", 1.5e-5)
        lstm_hidden = params.get("lstm_hidden", 384)
        dropout_rate = params.get("dropout_rate", 0.4)
        final_model = EfficientNetV2L_EnhancedBiLSTM(lstm_hidden=lstm_hidden, 
                                                     dropout_rate=dropout_rate).to(device)
        # Allow full fine-tuning in final training
        for param in final_model.backbone.parameters():
            param.requires_grad = True
            
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_loss = progressive_train_model(final_model, total_epochs, lr, final_checkpoint, batch_size,
                                             patience=5)
    else:
        print("\n--- Skipping Final Training (Checkpoint Exists) ---")
        print(f"Using existing model from: {final_checkpoint}")
    
    # ------------------------------
    # Evaluation on Test Set using Highest Resolution (300x300)
    # ------------------------------
    test_transform = get_transform(300)
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZES[300], shuffle=False, num_workers=2, pin_memory=True)
    
    # Load the best parameters from the tuning
    lstm_hidden = best_trial.params.get("lstm_hidden", 384)
    dropout_rate = best_trial.params.get("dropout_rate", 0.4)
    
    eval_model = EfficientNetV2L_EnhancedBiLSTM(lstm_hidden=lstm_hidden, 
                                               dropout_rate=dropout_rate).to(device)
    state = torch.load(final_checkpoint, map_location=device)
    eval_model.load_state_dict(state["model_state_dict"])
    eval_model.to(device)
    evaluate_model(eval_model, test_loader)
    torch.cuda.empty_cache()
    gc.collect()
    print("\n--- Evaluation Complete ---")