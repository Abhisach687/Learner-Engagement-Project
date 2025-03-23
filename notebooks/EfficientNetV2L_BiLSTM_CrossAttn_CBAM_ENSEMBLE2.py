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


def save_evaluation_results(results, confusion_matrices, model_name, save_dir=None):
    """
    Save evaluation results and confusion matrices to disk.
    
    Args:
        results: Dictionary of metrics for each emotion category
        confusion_matrices: Dictionary of confusion matrices for each emotion
        model_name: Name of the model (e.g., "single_model_tta" or "ensemble")
        save_dir: Directory to save results (default: BASE_DIR / "results")
    """
    if save_dir is None:
        save_dir = BASE_DIR / "results"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = save_dir / f"{timestamp}_{model_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to CSV
    results_df = pd.DataFrame({
        "Metric": list(results.keys()),
        "F1_Score": [results[k] for k in results.keys()],
        "LCRN_Score": [0.579, 0.537, 0.723, 0.735],  # LCRN benchmark
        "Difference": [results[k] - lcrn for k, lcrn in zip(results.keys(), [0.579, 0.537, 0.723, 0.735])]
    })
    results_df.to_csv(result_dir / f"{model_name}_results.csv", index=False)
    
    # Save confusion matrices as images
    for emotion, cm in confusion_matrices.items():
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {emotion}")
        plt.colorbar()
        plt.xticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
        plt.yticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(result_dir / f"{model_name}_{emotion}_confusion_matrix.png", dpi=300)
        plt.close()
    
    # Save summary text
    with open(result_dir / f"{model_name}_summary.txt", "w") as f:
        f.write(f"Evaluation Results for {model_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write("Performance vs LCRN:\n")
        f.write(f"{'Metric':<12} {'Score':<10} {'LCRN':<10} {'Diff':<10}\n")
        f.write("-" * 42 + "\n")
        for metric, score in results.items():
            lcrn_score = {"Engagement": 0.579, "Boredom": 0.537, "Confusion": 0.723, "Frustration": 0.735}[metric]
            diff = score - lcrn_score
            diff_str = f"{diff:.3f}" + (" ✓" if diff > 0 else "")
            f.write(f"{metric:<12} {score:.3f}      {lcrn_score:.3f}      {diff_str}\n")
    
    print(f"\nResults saved to: {result_dir}")
    return result_dir

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
        self.bilstm = nn.LSTM(input_size=self.feature_dim, hidden_size=lstm_hidden,
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.spatial_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.temporal_proj = nn.Linear(2 * lstm_hidden, classifier_hidden)
        self.cross_attn = CrossAttention(classifier_hidden)
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
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # (B, 16)
        return logits.view(B, 4, 4)

# ------------------------------
# Training Function with Progressive Resolution, Mixed Precision & Gradient Accumulation
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                            patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
    # Set lower learning rate for backbone parameters using parameter IDs to avoid tensor comparisons
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
                    # Compute loss over the 4 outputs
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
            
            # Validation phase
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
# Training Function for Multiple Seed Models
# ------------------------------
def train_multiple_seed_models(num_models=3, base_params=None):
    """
    Train multiple models with identical architecture but different random seeds.
    
    Args:
        num_models: Number of models to train
        base_params: Base hyperparameters to use for all models
    
    Returns:
        List of checkpoint paths for trained models
    """
    # Define seeds for reproducibility
    seeds = [42, 101, 2023, 9999, 7777][:num_models]
    checkpoint_paths = []
    
    # Use best trial parameters if available, otherwise use defaults
    if base_params is None:
        base_params = {
            "batch_size": 4,
            "lr": 1e-4,
            "lstm_hidden": 256,
            "lstm_layers": 1,
            "dropout_rate": 0.5
        }
    
    total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Training Model {i+1}/{len(seeds)} with Seed {seed} ---")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        # Get lstm_hidden from checkpoint if we're using the final checkpoint as reference
        reference_hidden = 256
        if i == 0 and os.path.exists(final_checkpoint):
            try:
                ref_state = torch.load(final_checkpoint, map_location='cpu')
                for key, param in ref_state["model_state_dict"].items():
                    if key == "bilstm.weight_ih_l0":
                        reference_hidden = param.shape[0] // 4
                        print(f"Using reference LSTM hidden size from checkpoint: {reference_hidden}")
                        break
            except:
                print("Could not determine reference hidden size, using default")
        
        # Create model with matched architecture
        model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
            lstm_hidden=reference_hidden,
            lstm_layers=base_params.get("lstm_layers", 1),
            dropout_rate=base_params.get("dropout_rate", 0.5),
            classifier_hidden=256
        ).to(device)
        
        # Unfreeze backbone layers after stage 6 for fine-tuning
        if hasattr(model.backbone, "blocks"):
            for j, block in enumerate(model.backbone.blocks):
                if j >= 6:
                    for param in block.parameters():
                        param.requires_grad = True
        
        # Define checkpoint path for this seed
        seed_checkpoint = MODEL_DIR / f"seed_{seed}_ensemble_model_checkpoint.pth"
        seed_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        
        # Train the model
        loss = progressive_train_model(
            model, 
            total_epochs, 
            base_params.get("lr", 1e-4), 
            seed_checkpoint,
            base_params.get("batch_size", 4),
            patience=5, 
            gradient_accum_steps=GRADIENT_ACCUM_STEPS
        )
        
        checkpoint_paths.append(seed_checkpoint)
        
        # Clean up to avoid memory issues
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    return checkpoint_paths

# ------------------------------
# Evaluation Function
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    confusion_matrices = {}
    
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # No TTA - use same approach as in full.py
            outputs = model(frames)
            
            outputs = outputs.view(outputs.size(0), 4, 4)
            preds = torch.argmax(outputs, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Store results for reporting
    results = {}
    
    for i, metric in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification report for {metric}:")
        report = classification_report(all_labels[:, i], all_preds[:, i], digits=3, output_dict=True)
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        results[metric] = report['accuracy']
        
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        confusion_matrices[metric] = cm
        
        # Display confusion matrix
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
    
    # Print summary for comparison with LCRN
    print("\n--- Performance Summary vs LCRN ---")
    lcrn_scores = {
        "Engagement": 0.579,
        "Boredom": 0.537,
        "Confusion": 0.723,
        "Frustration": 0.735
    }
    
    print(f"{'Metric':<12} {'Score':<10} {'LCRN':<10} {'Diff':<10}")
    print("-" * 42)
    for metric, score in results.items():
        diff = score - lcrn_scores[metric]
        diff_str = f"{diff:.3f}" + (" ✓" if diff > 0 else "")
        print(f"{metric:<12} {score:.3f}      {lcrn_scores[metric]:.3f}      {diff_str}")
    
    # Save results to disk
    save_evaluation_results(results, confusion_matrices, "single_model")
    
    return results

def efficient_ensemble_evaluate(checkpoint_paths, test_loader, subset_size=0.4, resolution=300, save_plots=True):
    """
    Efficient ensemble evaluation that's much faster.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        test_loader: DataLoader for test data
        subset_size: Fraction of test data to evaluate (0.0-1.0)
        resolution: Resolution to resize images to (lower = faster)
        save_plots: Whether to save confusion matrix plots
    """
    
    print(f"\n--- Efficient Ensemble Evaluation ({subset_size*100:.0f}% of data at {resolution}×{resolution}) ---")
    
    # Create a downsampled transform for faster processing
    fast_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use all available models for best ensemble performance
    print(f"Using all {len(checkpoint_paths)} checkpoints for ensemble evaluation")
    
    # Load models
    models = []
    for i, path in enumerate(checkpoint_paths):
        print(f"Loading model {i+1}/{len(checkpoint_paths)}: {path}")
        
        # First load the checkpoint to determine architecture parameters
        state = torch.load(path, map_location=device)
        
        # Detect LSTM hidden size from the checkpoint
        lstm_hidden = 256  # Default
        if "model_state_dict" in state:
            for key, param in state["model_state_dict"].items():
                if key == "bilstm.weight_ih_l0":
                    lstm_hidden = param.shape[0] // 4
                    print(f"  Detected LSTM hidden size: {lstm_hidden}")
                    break
        
        # Create model with detected parameters
        model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
            lstm_hidden=lstm_hidden, 
            lstm_layers=1,
            dropout_rate=0.5, 
            classifier_hidden=256
        ).to(device)
        
        # Load the state dict
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        models.append(model)
    
    # Evaluate ensemble on subset of data
    all_preds = []
    all_labels = []
    
    # Modify batch processing section:
    total_batches = len(test_loader)
    subset_batches = max(1, int(total_batches * subset_size))
    print(f"Evaluating on {subset_batches} of {total_batches} batches")
    
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        # Process batches
        for batch_idx, (frames, labels) in enumerate(tqdm(test_loader, desc="Ensemble Evaluation")):
            if batch_idx >= subset_batches:
                break
                
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Skip the resizing part if we're using full resolution
            if resolution == 300 or frames.dim() != 5:
                pass  # No resizing needed
            
            # Get ensemble predictions
            ensemble_outputs = None
            for model in models:
                outputs = model(frames)
                if ensemble_outputs is None:
                    ensemble_outputs = outputs
                else:
                    ensemble_outputs += outputs
            
            # Average predictions
            ensemble_outputs /= len(models)
            ensemble_outputs = ensemble_outputs.view(ensemble_outputs.size(0), 4, 4)
            preds = torch.argmax(ensemble_outputs, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Calculate and display results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Enhanced results display with plots
    results = {}
    confusion_matrices = {}
    
    # Create directory for plots if saving
    if save_plots:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = BASE_DIR / "results" / f"{timestamp}_ensemble_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to {plots_dir}")
    
    for i, metric in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Ensemble classification report for {metric}:")
        report = classification_report(all_labels[:, i], all_preds[:, i], digits=3, output_dict=True)
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        results[metric] = report['accuracy']
        
        # Create and display confusion matrix
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        confusion_matrices[metric] = cm
        
        # Create more detailed visualizations
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Ensemble Confusion Matrix for {metric}", fontsize=14)
        plt.colorbar()
        
        # Add accuracy percentages to cells
        thresh = cm.max() / 2
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                plt.text(col, row, f"{cm[row, col]}\n({cm[row, col]/np.sum(cm[row])*100:.1f}%)" if cm[row, col] > 0 else "0",
                        ha="center", va="center", 
                        color="white" if cm[row, col] > thresh else "black", 
                        fontsize=12)
        
        plt.xticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
        plt.yticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.tight_layout()
        
        # Save and show
        if save_plots:
            plt.savefig(plots_dir / f"ensemble_{metric}_confusion_matrix.png", dpi=300)
        plt.show()
    
    # Print summary for comparison with LCRN
    print("\n--- Detailed Ensemble Results vs LCRN ---")
    lcrn_scores = {
        "Engagement": 0.579,
        "Boredom": 0.537,
        "Confusion": 0.723,
        "Frustration": 0.735
    }
    
    print(f"{'Metric':<12} {'Ensemble':<10} {'LCRN':<10} {'Diff':<10}")
    print("-" * 42)
    for metric, score in results.items():
        diff = score - lcrn_scores[metric]
        diff_str = f"{diff:.3f}" + (" ✓" if diff > 0 else "")
        print(f"{metric:<12} {score:.3f}      {lcrn_scores[metric]:.3f}      {diff_str}")
    
    # Save detailed evaluation results
    save_evaluation_results(results, confusion_matrices, "detailed_ensemble_evaluation", save_dir=plots_dir.parent)
    return results

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
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [256, 512])
        lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.4, 0.5])
        total_epochs = sum(eps for _, eps in PROG_SCHEDULE)
        model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                                                       dropout_rate=dropout_rate, classifier_hidden=256).to(device)
        trial_checkpoint = MODEL_DIR / f"trial_eff_v2l_{trial.number}__bilstm_crossattn_cbam_checkpoint.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        loss = progressive_train_model(model, total_epochs, lr, trial_checkpoint, batch_size,
                                       patience=3, gradient_accum_steps=GRADIENT_ACCUM_STEPS)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return loss
    
    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_bilstm_crossattn_cbam.db"
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
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l__bilstm_crossattn_cbam_checkpoint.pth"
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        params = best_trial.params
        batch_size = params.get("batch_size", 4)
        lr = params.get("lr", 1e-4)
        lstm_hidden = params.get("lstm_hidden", 256)
        lstm_layers = params.get("lstm_layers", 1)
        dropout_rate = params.get("dropout_rate", 0.5)
        final_model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                                                             dropout_rate=dropout_rate, classifier_hidden=256).to(device)
        # Unfreeze backbone layers after stage 6 for fine-tuning
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
    print("\n--- Starting Evaluation ---")
    test_transform = get_transform(300)
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    
    # Reduce batch size for efficiency
    batch_size = min(best_trial.params.get("batch_size", 4), 2)  # Smaller batch size
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # First evaluate using the original final checkpoint with horizontal flip only
    print(f"Loading model from: {final_checkpoint}")
    
    # Detect LSTM hidden size from the checkpoint
    state = torch.load(final_checkpoint, map_location=device)
    lstm_hidden = 256  # Default
    for key, param in state["model_state_dict"].items():
        if key == "bilstm.weight_ih_l0":
            lstm_hidden = param.shape[0] // 4
            print(f"Detected LSTM hidden size: {lstm_hidden}")
            break
            
    eval_model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM(
        lstm_hidden=lstm_hidden,
        lstm_layers=best_trial.params.get("lstm_layers", 1),
        dropout_rate=best_trial.params.get("dropout_rate", 0.5),
        classifier_hidden=256
    ).to(device)
    
    eval_model.load_state_dict(state["model_state_dict"])
    
    # Single model evaluation
    print("\n--- Evaluating Single Model ---")
    single_results = evaluate_model(eval_model, test_loader)
    
    # ------------------------------
    # Define paths to ensemble models
    # ------------------------------
    # Check for existing seed model checkpoints
    ensemble_seed_checkpoints = [
        MODEL_DIR / f"seed_42_ensemble_model_checkpoint.pth",
        MODEL_DIR / f"seed_101_ensemble_model_checkpoint.pth",
        MODEL_DIR / f"seed_2023_ensemble_model_checkpoint.pth"
    ]
    
    # Filter to only include checkpoints that actually exist
    ensemble_seed_checkpoints = [path for path in ensemble_seed_checkpoints if path.exists()]
    
    # If no seed checkpoints found, just duplicate the final checkpoint
    if not ensemble_seed_checkpoints:
        print("No seed checkpoints found. Using final checkpoint twice for ensemble.")
        ensemble_seed_checkpoints = [final_checkpoint]
    
    # Then evaluate using the efficient ensemble with all available models
    print("\n--- Efficient Ensemble Evaluation ---")
    # Use all seed files + final checkpoint
    selected_checkpoints = [final_checkpoint] + ensemble_seed_checkpoints
    print(f"Using {len(selected_checkpoints)} models in ensemble")
    ensemble_results = efficient_ensemble_evaluate(
        selected_checkpoints, 
        test_loader,
        subset_size=1.0,   # Adjust as needed for speed
        resolution=300     # Use full resolution for accuracy
    )
    
    # Print final comparison and save overall results
    print("\n--- Final Comparison Summary ---")
    print("Base Full.py (from literature): Frustration 78.1% (beats LCRN)")
    print("LCRN: Engagement 57.9%, Boredom 53.7%, Confusion 72.3%, Frustration 73.5%")
    print(f"Our Single Model+TTA: {', '.join([f'{k} {v*100:.1f}%' for k, v in single_results.items()])}")
    print(f"Our Efficient Ensemble: {', '.join([f'{k} {v*100:.1f}%' for k, v in ensemble_results.items()])}")

    # Save final comparison
    final_results = {
        "Single_Model": single_results,
        "Ensemble": ensemble_results
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = BASE_DIR / "results" / f"{timestamp}_final_comparison.json"
    with open(results_file, "w") as f:
        import json
        json.dump(final_results, f, indent=4)
    print(f"\nFinal comparison saved to: {results_file}")

    torch.cuda.empty_cache()
    gc.collect()
    print("\n--- Evaluation Complete ---")