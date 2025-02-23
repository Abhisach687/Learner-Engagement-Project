import os
import cv2
import gc
import torch
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
import matplotlib.pyplot as plt
import timm  # pip install timm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONFIG & HYPERPARAMETERS (SAMPLE MODE)
# ------------------------------
NUM_FRAMES = 3  # For quick testing, we use 3 frames per video
# Progressive resolution schedule: (resolution, epochs) - minimal for testing
PROG_SCHEDULE = [(112, 1), (224, 1), (300, 1)]
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
GRADIENT_ACCUM_STEPS = 1  # no accumulation for sample

# ------------------------------
# Setup directories for sample data (using a "sample_data" folder)
# ------------------------------
BASE_DIR = Path("sample_data")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

for d in [FRAMES_DIR, LABELS_DIR, MODEL_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    # For sample, we simply return the base without any modifications.
    return base

def select_impactful_frames(video_folder: Path, num_frames=NUM_FRAMES):
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
def precompute_best_frames(csv_file: Path, video_root: Path, num_frames=NUM_FRAMES, resolution=224):
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

def convert_pkl_to_lmdb(csv_file: Path, num_frames=NUM_FRAMES, resolution=224,
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
# LMDB Dataset Class
# ------------------------------
class VideoDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self, csv_file, lmdb_path, num_frames=NUM_FRAMES, resolution=224):
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

# ------------------------------
# Dummy Model Components
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
        query = self.query(spatial_feat).unsqueeze(1)
        key = self.key(temporal_feat).unsqueeze(2)
        attn = self.softmax(torch.bmm(query, key))
        value = self.value(temporal_feat)
        return attn.squeeze(-1) * value

# ------------------------------
# Model Architecture (Simplified Sample Version)
# ------------------------------
class EfficientNetV2L_BiLSTM_CrossAttn_CBAM(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=1, dropout_rate=0.5, classifier_hidden=256):
        super(EfficientNetV2L_BiLSTM_CrossAttn_CBAM, self).__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        # Freeze early layers
        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < 6:
                    for param in block.parameters():
                        param.requires_grad = False
        # Dynamically set feature_dim from the backbone (using num_features if available)
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = 1536  # fallback
        self.cbam = CBAM(self.feature_dim, reduction=16, kernel_size=7)
        self.bilstm = nn.LSTM(input_size=self.feature_dim, hidden_size=lstm_hidden,
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.spatial_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.temporal_proj = nn.Linear(2 * lstm_hidden, classifier_hidden)
        self.cross_attn = CrossAttention(classifier_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(classifier_hidden * 2, 16)
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        features = self.backbone(x)  # expected shape: (B*T, feature_dim)
        features = features.view(B, T, self.feature_dim)
        features = features.permute(0, 2, 1)
        features = self.cbam(features)
        features = features.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.bilstm(features)
        temporal_context = torch.cat((h_n[-2], h_n[-1]), dim=1)
        spatial_feature = features.mean(dim=1)
        spatial_proj = self.spatial_proj(spatial_feature)
        temporal_proj = self.temporal_proj(temporal_context)
        cross_context = self.cross_attn(spatial_proj, temporal_proj)
        fusion = torch.cat((spatial_proj, cross_context), dim=1)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)
        return logits.view(B, 4, 4)

# ------------------------------
# Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Dummy Data Creation for Testing
# ------------------------------
def create_dummy_csv(split: str):
    df = pd.DataFrame({
        "ClipID": ["dummy1.jpg"],
        "Engagement": [0],
        "Boredom": [0],
        "Confusion": [0],
        "Frustration": [0]
    })
    csv_path = LABELS_DIR / f"{split}Labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def create_dummy_video_frames(split: str):
    folder = FRAMES_DIR / split / "dummy1"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, NUM_FRAMES + 1):
        img = Image.new('RGB', (100, 100), color=(i*40 % 255, i*80 % 255, i*120 % 255))
        img_path = folder / f"frame_{i:04d}.jpg"
        img.save(img_path)

# ------------------------------
# Main execution (sample testing)
# ------------------------------
def main():
    for split in ["Train", "Validation", "Test"]:
        create_dummy_csv(split)
        create_dummy_video_frames(split)

    resolutions = [112, 224, 300]
    csv_files = {
        "Train": LABELS_DIR / "TrainLabels.csv",
        "Validation": LABELS_DIR / "ValidationLabels.csv",
        "Test": LABELS_DIR / "TestLabels.csv"
    }
    for res in resolutions:
        for split, csv_file in csv_files.items():
            print(f"\nProcessing {csv_file} at resolution {res}")
            precompute_best_frames(csv_file, FRAMES_DIR, num_frames=NUM_FRAMES, resolution=res)
            convert_pkl_to_lmdb(csv_file, num_frames=NUM_FRAMES, resolution=res,
                                transform=get_transform(res), lmdb_map_size=1 * 1024**3)

    # For sample testing, load the Train LMDB at resolution 112
    train_lmdb = CACHE_DIR / f"lmdb_TrainLabels_frame_{NUM_FRAMES}_112"
    train_set = VideoDatasetLMDB(csv_files["Train"], train_lmdb, num_frames=NUM_FRAMES, resolution=112)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    # Instantiate the model
    model = EfficientNetV2L_BiLSTM_CrossAttn_CBAM().to(device)
    model.eval()

    for features, labels in train_loader:
        print("Features shape from LMDB sample:", features.shape)
        # Create a dummy input mimicking raw images with shape (B, T, 3, 112, 112)
        B, T, feat_dim = features.shape
        dummy_input = torch.randn(B, T, 3, 112, 112).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
        print("Model output shape:", outputs.shape)
        break

    print("\nSample run successful. Pipeline functions as expected.")

if __name__ == "__main__":
    main()
