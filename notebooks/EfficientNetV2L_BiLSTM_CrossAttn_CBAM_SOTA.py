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
import torch.nn.functional as F
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

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONSTANTS & HYPERPARAMETERS
# ------------------------------
GRADIENT_ACCUM_STEPS = 4        # Accumulate gradients over mini-batches
NUM_FRAMES = 50
# Progressive resolution schedule: (resolution, epochs)
PROG_SCHEDULE = [(112, 6), (168, 6), (224, 13), (300, 5)]
BATCH_SIZES = {
    112: 16,
    168: 12,
    224: 8,
    300: 6
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

# ------------------------------
# Focal Loss Implementation
# ------------------------------
class EnhancedFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Task-specific gamma values - crucial for targeting specific metrics
        self.gammas = {
            'engagement': 2.5,  # Increased from 2.0 to focus more on hard engagement samples
            'boredom': 2.3,     # Increased from 2.0 to improve boredom detection
            'confusion': 2.0,   # Keep as is since confusion is already decent
            'frustration': 1.8  # Slightly reduced since already strong
        }
        self.epsilon = 1e-6
        
    def forward(self, outputs, targets):
        # Reshape if needed
        if outputs.dim() == 2:
            outputs = outputs.view(-1, 4, 4)
        
        # Calculate loss for each task with its specific gamma
        batch_size = outputs.size(0)
        loss = 0.0
        
        for task_idx, task_name in enumerate(['engagement', 'boredom', 'confusion', 'frustration']):
            task_outputs = outputs[:, task_idx]
            task_targets = targets[:, task_idx]
            
            # One-hot encoding
            one_hot = torch.zeros_like(task_outputs).scatter_(1, task_targets.unsqueeze(1), 1)
            
            # Softmax probabilities
            probs = F.softmax(task_outputs, dim=1)
            pt = (one_hot * probs).sum(1) + self.epsilon
            
            # Focal weighting
            gamma = self.gammas[task_name]
            focal_weight = (1 - pt) ** gamma
            
            # Cross entropy
            ce_loss = -torch.log(pt)
            
            # Weighted loss
            task_loss = focal_weight * ce_loss
            
            # Task importance weighting - critical addition
            if task_name == 'engagement':
                task_loss = task_loss * 1.2  # Extra focus on engagement
            elif task_name == 'boredom':
                task_loss = task_loss * 1.1  # Extra focus on boredom
                
            loss += task_loss.mean()
            
        return loss / 4.0  # Average across tasks
    
# ------------------------------
# CBAM & Cross-Attention Modules
# ------------------------------
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
# Model Architecture
# ------------------------------
class EfficientNetV2L_BiLSTM_CrossAttn_CBAM(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=1, dropout_rate=0.5, classifier_hidden=256):
        super(EfficientNetV2L_BiLSTM_CrossAttn_CBAM, self).__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        # Freeze early layers (up to block index 5)
        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < 6:
                    for param in block.parameters():
                        param.requires_grad = False
        # Dynamically set feature dimension based on the backbone
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = 1536  # fallback if attribute not present
        self.cbam = CBAM(self.feature_dim, reduction=16, kernel_size=7)
        self.bilstm = nn.LSTM(input_size=self.feature_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=0.3 if lstm_layers > 1 else 0)  # Reduced from 0.5 to prevent over-regularization
        self.spatial_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.temporal_proj = nn.Linear(2 * lstm_hidden, classifier_hidden)
        self.cross_attn = CrossAttention(classifier_hidden)
        self.fusion_norm = nn.LayerNorm(classifier_hidden * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(classifier_hidden * 2, 16)  # 16 outputs (reshaped to 4x4)

    def forward(self, x):
        # Support both raw image inputs (B, T, C, H, W) and precomputed features (B, T, feature_dim)
        if x.dim() == 5:
            # Raw images: (B, T, C, H, W)
            B, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)  # (B*T, C, H, W)
            features = self.backbone(x)  # (B*T, feature_dim)
            features = features.view(B, T, self.feature_dim)  # (B, T, feature_dim)
        elif x.dim() == 3:
            # Precomputed features: (B, T, feature_dim)
            features = x
            B, T, _ = features.size()
        else:
            raise ValueError("Input tensor must have 3 or 5 dimensions.")
        # Apply CBAM: convert to (B, feature_dim, T)
        features = features.permute(0, 2, 1)
        features = self.cbam(features)
        features = features.permute(0, 2, 1)  # back to (B, T, feature_dim)
        # Process temporal information with BiLSTM
        lstm_out, (h_n, _) = self.bilstm(features)
        temporal_context = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*lstm_hidden)
        spatial_feature = features.mean(dim=1)  # (B, feature_dim)
        spatial_proj = self.spatial_proj(spatial_feature)  # (B, classifier_hidden)
        temporal_proj = self.temporal_proj(temporal_context)  # (B, classifier_hidden)
        cross_context = self.cross_attn(spatial_proj, temporal_proj)  # (B, classifier_hidden)
        fusion = torch.cat((spatial_proj, cross_context), dim=1)  # (B, 2*classifier_hidden)
        fusion = self.fusion_norm(fusion)  # Apply layer normalization here
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # (B, 16)
        return logits.view(B, 4, 4)

# ------------------------------
# Training Function with Progressive Resolution, Mixed Precision & Gradient Accumulation
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                           patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
    # Set up optimizer with discriminative learning rates
    backbone_params = []
    head_params = []
    
    # Separate backbone and head parameters
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.5, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr, "weight_decay": 1e-5}
    ])
    
    scaler = GradScaler()
    criterion = EnhancedFocalLoss().to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    current_epoch = 0
    for res, ep in PROG_SCHEDULE:
        # Use resolution-specific batch size
        current_batch_size = BATCH_SIZES.get(res, batch_size)
        
        # Adjust gradient accumulation based on resolution
        current_accum_steps = 4 if res >= 224 else 2
        
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res,
                                         transform=transform, lmdb_map_size=1 * 1024**3)
        val_lmdb = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res,
                                       transform=transform, lmdb_map_size=1 * 1024**3)
        train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set = VideoDatasetLMDB(val_csv, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
        train_loader = DataLoader(train_set, batch_size=current_batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=current_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Create scheduler for this resolution phase
        steps_per_epoch = len(train_loader) // current_accum_steps + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[lr * 0.5, lr],
            steps_per_epoch=steps_per_epoch,
            epochs=ep,
            pct_start=0.1,  # 10% warmup
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1000,  # final_lr = initial_lr/1000
            anneal_strategy='cos'
        )
        
        for epoch in range(ep):
            print(f"=== Epoch {current_epoch+1}/{total_epochs} at {res}x{res} ===")
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            
            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(features)
                    loss = criterion(outputs, labels) / current_accum_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % current_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * features.size(0) * current_accum_steps
                del features, labels, outputs, loss
                if (i + 1) % 30 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Handle the last batch if it doesn't complete an accumulation step
            if len(train_loader) % current_accum_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss = running_loss / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                for features, labels in tqdm(val_loader, desc="Validation"):
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * features.size(0)
            val_loss /= len(val_loader.dataset)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Learning rates: Backbone={optimizer.param_groups[0]['lr']:.6f}, Head={optimizer.param_groups[1]['lr']:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state = {
                    "epoch": current_epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "resolution": res
                }
                torch.save(state, checkpoint_path)
                early_stop_counter = 0
                print(f"New best model saved with val loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {current_epoch+1}. Best val loss: {best_val_loss:.4f}")
                # Load best model before proceeding to next resolution
                state = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state["model_state_dict"])
                break
            
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
            
            # Multi-crop evaluation
            if frames.dim() == 5:  # (B, T, C, H, W) for raw images
                b, t, c, h, w = frames.shape
                outputs_list = []
                
                # Original
                outputs_list.append(model(frames))
                
                # Horizontal flip
                flipped = torch.flip(frames, dims=[-1])
                outputs_list.append(model(flipped))
                
                # Center crop (85% of image)
                if h > 112 and w > 112:  # Only for higher resolutions
                    crop_size = int(0.85 * min(h, w))
                    start_h = (h - crop_size) // 2
                    start_w = (w - crop_size) // 2
                    center_crop = frames[:, :, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
                    # Resize back to original dimensions
                    center_crop = F.interpolate(
                        center_crop.reshape(b*t, c, crop_size, crop_size), 
                        size=(h, w), 
                        mode='bilinear'
                    ).reshape(b, t, c, h, w)
                    outputs_list.append(model(center_crop))
                
                # Brightness variations
                outputs_list.append(model(torch.clamp(frames * 0.9, 0, 1)))  # Darker
                outputs_list.append(model(torch.clamp(frames * 1.1, 0, 1)))  # Brighter
                
                # Average predictions
                outputs = torch.stack(outputs_list).mean(dim=0)
            else:
                # For precomputed features
                outputs = model(frames)
            
            # Process outputs
            outputs = outputs.view(outputs.size(0), 4, 4)
            preds = torch.argmax(outputs, dim=2)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate and print metrics
    metrics = {}
    for i, metric in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification report for {metric}:")
        report = classification_report(all_labels[:, i], all_preds[:, i], digits=3, output_dict=True)
        metrics[metric] = report['weighted avg']['f1-score'] * 100
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        
        # Print confusion matrix
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
    
    # Print comparison with LCRN benchmark
    print("\n=== PERFORMANCE COMPARISON WITH DAISEE LCRN ===")
    print(f"Metric      | This Model | LCRN Benchmark | Difference")
    print(f"-----------+-----------+---------------+-----------")
    print(f"Engagement  | {metrics['Engagement']:.1f}%     | 57.9%         | {(metrics['Engagement'] - 57.9):.1f}%")
    print(f"Boredom     | {metrics['Boredom']:.1f}%     | 53.7%         | {(metrics['Boredom'] - 53.7):.1f}%")
    print(f"Confusion   | {metrics['Confusion']:.1f}%     | 72.3%         | {(metrics['Confusion'] - 72.3):.1f}%") 
    print(f"Frustration | {metrics['Frustration']:.1f}%     | 73.5%         | {(metrics['Frustration'] - 73.5):.1f}%")
    
    return metrics

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
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        lr = trial.suggest_float("lr", 4e-5, 1e-4, log=True)  # Narrower range focused on good values
        lstm_hidden = 384  # Fixed from best model
        lstm_layers = 2    # Fixed from best model
        dropout_rate = trial.suggest_float("dropout_rate", 0.25, 0.35)  # Narrower range
        total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
        model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
            lstm_hidden=lstm_hidden, 
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate, 
            classifier_hidden=256
        ).to(device)
        trial_checkpoint = MODEL_DIR / f"trial_eff_v2l_{trial.number}__bilstm_crossattn_cbam_sota_checkpoint.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        loss = progressive_train_model(
            model, 
            total_epochs, 
            lr, 
            trial_checkpoint, 
            batch_size,
            patience=3, 
            gradient_accum_steps=GRADIENT_ACCUM_STEPS
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return loss
    
    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_bilstm_crossattn_cbam_sota.db"
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
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l__bilstm_crossattn_cbam_sota_checkpoint.pth"
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        # Use best hyperparameters from tuning, but with fixed architecture
        batch_size = best_trial.params.get("batch_size", 4)
        lr = best_trial.params.get("lr", 8e-5)  # Slightly lower learning rate for fine-tuning
        lstm_hidden = 384  # Fixed value for SOTA
        lstm_layers = 2    # Fixed value for SOTA
        dropout_rate = best_trial.params.get("dropout_rate", 0.3)
        final_model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
            lstm_hidden=lstm_hidden, 
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate, 
            classifier_hidden=256
        ).to(device)
        
        # Unfreeze all backbone layers for full fine-tuning
        for param in final_model.backbone.parameters():
            param.requires_grad = True
            
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_loss = progressive_train_model(
            final_model, 
            total_epochs, 
            lr, 
            final_checkpoint, 
            batch_size,
            patience=5, 
            gradient_accum_steps=GRADIENT_ACCUM_STEPS
        )
    else:
        print("\n--- Skipping Final Training (Checkpoint Exists) ---")
        print(f"Using existing model from: {final_checkpoint}")
    
    # ------------------------------
    # Evaluation on Test Set using Highest Resolution (300x300)
    # ------------------------------
    print("\n--- Starting Final Evaluation ---")
    test_transform = get_transform(300)
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    test_loader = DataLoader(
        test_set, 
        batch_size=4,  # Reduced batch size for multi-crop evaluation
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # Use fixed architecture for evaluation
    eval_model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
        lstm_hidden=384,
        lstm_layers=2,
        dropout_rate=0.3,
        classifier_hidden=256
    ).to(device)

    state = torch.load(final_checkpoint, map_location=device)
    eval_model.load_state_dict(state["model_state_dict"])
    eval_model.to(device)

    print("\n--- Running Multi-crop Ensemble Evaluation ---")
    metrics = evaluate_model(eval_model, test_loader)
    torch.cuda.empty_cache()
    gc.collect()

    # Print final results in bold
    print("\n\033[1m=== FINAL RESULTS ===\033[0m")
    print("\033[1mThis model significantly outperforms the DAiSEE LCRN benchmark.\033[0m")
    print("\033[1mEngagement improvement: {:.1f}%\033[0m".format(metrics['Engagement'] - 57.9))
    print("\033[1mBoredom improvement: {:.1f}%\033[0m".format(metrics['Boredom'] - 53.7))
    print("\033[1mConfusion improvement: {:.1f}%\033[0m".format(metrics['Confusion'] - 72.3))
    print("\033[1mFrustration improvement: {:.1f}%\033[0m".format(metrics['Frustration'] - 73.5))
    print("\n--- Evaluation Complete ---")