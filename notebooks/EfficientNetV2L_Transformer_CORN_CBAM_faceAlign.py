import os
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
from torch.utils.data.sampler import WeightedRandomSampler
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import timm  # pip install timm
import urllib.request
from torch.utils.checkpoint import checkpoint

logging.basicConfig(level=logging.INFO)



# ------------------------------
# Fallback for coral_pytorch
# ------------------------------
try:
    from coral_pytorch.layers import CoralLayer as ImportedCoralLayer
    from coral_pytorch.losses import corn_loss as imported_corn_loss
    _ = ImportedCoralLayer(10, 4)  # Test constructor with expected parameters
    CoralLayer = ImportedCoralLayer
    corn_loss = imported_corn_loss
except Exception as e:
    print(f"Error importing CORAL components: {e}")
    logging.warning(f"Using fallback implementations for CoralLayer and corn_loss. Error: {e}")
    class CoralLayer(nn.Module):
        def __init__(self, num_features, num_classes, **kwargs):
            super().__init__()
            self.linear = nn.Linear(num_features, num_classes - 1)
        def forward(self, x):
            return self.linear(x)
    def corn_loss(logits, targets, num_classes=None):
        if num_classes is None:
            num_classes = logits.size(1) + 1
        bce = nn.BCEWithLogitsLoss()
        target_matrix = torch.zeros_like(logits, dtype=torch.float)
        for k in range(1, num_classes):
            target_matrix[:, k - 1] = (targets >= k).float()
        return bce(logits, target_matrix)

# ------------------------------
# New: SpatialDropout Module (using Dropout1d)
# ------------------------------
class SpatialDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout1d = nn.Dropout1d(p)
    def forward(self, x):
        # x shape: (B, C, T)
        return self.dropout1d(x)

# ------------------------------
# Hyperparameters & Constants
# ------------------------------
GRADIENT_ACCUM_STEPS = 4        # Grad accumulation steps
NUM_FRAMES = 50
PROG_SCHEDULE = [(112, 5), (224, 10), (300, 15)]
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
NUM_DAISEE_CLASSES = 4

temporal_model_type = "bilstm"
classifier_fc_layers = 1
loss_type = "corn"

# Option for optimizer type and Lookahead
try:
    import torch_optimizer as toptim
    OPTIMIZER_OPTIONS = {"adamw": optim.AdamW, "radam": toptim.RAdam}
    USE_LOOKAHEAD = True
    LOOKAHEAD_K = 5
    LOOKAHEAD_ALPHA = 0.5
except ImportError:
    OPTIMIZER_OPTIONS = {"adamw": optim.AdamW}
    USE_LOOKAHEAD = False

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

train_csv = LABELS_DIR / "TrainLabels.csv"
val_csv = LABELS_DIR / "ValidationLabels.csv"
test_csv = LABELS_DIR / "TestLabels.csv"

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
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ------------------------------
# Temporal Augmentations (advanced)
# ------------------------------
def temporal_augment(frames, drop_prob=0.2, shuffle_prob=0.2):
    frames = list(frames)
    if len(frames) > 1 and np.random.rand() < drop_prob:
        drop_idx = np.random.randint(0, len(frames))
        frames.pop(drop_idx)
        frames.append(frames[-1])
    if len(frames) > 1 and np.random.rand() < shuffle_prob:
        idx = np.random.randint(0, len(frames)-1)
        frames[idx], frames[idx+1] = frames[idx+1], frames[idx]
    return frames

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
            frames = temporal_augment(frames)
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
    backbone.global_pool = nn.AdaptiveAvgPool2d(1)
    backbone.eval()
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_out = backbone(dummy_input).flatten(1)
    feature_dim = dummy_out.shape[1]
    print(f"Inferred backbone feature dimension: {feature_dim}")

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
                with torch.no_grad(), autocast(enabled=False):
                    feat = checkpoint(lambda x: backbone(x), tensor, use_reentrant=False)
                    feat = feat.flatten(1).squeeze(0).cpu().half().detach()
                video_features.append(feat.numpy())
            if video_features:
                video_features_np = np.stack(video_features)
                key = f"video_{valid_indices[idx]}".encode("utf-8")
                txn.put(key, pickle.dumps(video_features_np))
    env.close()
    print(f"LMDB database created at {lmdb_path}")
    return lmdb_path

# ------------------------------
# Dataset Classes
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
                size = self.transform.transforms[0].size if self.transform is not None and hasattr(self.transform.transforms[0], "size") else 224
                img = Image.new('RGB', (size, size))
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video_tensor = torch.stack(frames)
        labels = self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return video_tensor, torch.tensor(labels.values, dtype=torch.long)
    
# Create custom dataset focusing on difficult examples
class TargetedVideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_root, difficult_indices, num_frames=50, transform=None):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        self.video_root = video_root
        self.difficult_indices = difficult_indices
        self.num_frames = num_frames
        self.transform = transform
        self.valid_indices = []  # Keep track of valid indices
        self._validate_indices()  # Validate indices on initialization
        
    def _validate_indices(self):
        """Pre-check indices to filter out any that don't have valid frames"""
        valid_indices = []
        print("Validating difficult examples...")
        for idx in tqdm(self.difficult_indices):
            try:
                row = self.data.iloc[idx]
                clip_id = get_csv_clip_id(row["ClipID"].split('.')[0])
                video_folder = self.video_root / "Train" / clip_id
                frames = select_impactful_frames(video_folder, self.num_frames)
                if len(frames) >= 1:
                    valid_indices.append(idx)
            except Exception:
                pass  # Skip invalid indices
        self.difficult_indices = valid_indices
        print(f"Found {len(valid_indices)} valid examples out of {len(self.difficult_indices)} difficult examples")

    def __len__(self):
        return len(self.difficult_indices)

    def __getitem__(self, idx):
        original_idx = self.difficult_indices[idx]
        row = self.data.iloc[original_idx]
        clip_id = get_csv_clip_id(row["ClipID"].split('.')[0])
        video_folder = self.video_root / "Train" / clip_id
        frames = select_impactful_frames(video_folder, self.num_frames)
        
        # Safety check - should never happen due to validation, but just in case
        if len(frames) == 0:
            # Create a dummy frame if none exist
            size = 300 if self.transform is None else 300
            dummy_img = Image.new('RGB', (size, size))
            if self.transform:
                dummy_img = self.transform(dummy_img)
            video_tensor = torch.stack([dummy_img] * self.num_frames)
            labels = row[["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
            return video_tensor, torch.tensor(labels.values, dtype=torch.long)
        
        frames = temporal_augment(frames)
        
        # Handle the case where we don't have enough frames
        if len(frames) < self.num_frames:
            # Duplicate the last frame to fill up to num_frames
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))
        
        # Load images with error handling
        images = []
        for fp in frames:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                # Create a dummy image on error
                size = 300 if self.transform is None else 300
                img = Image.new('RGB', (size, size))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        video_tensor = torch.stack(images)
        labels = row[["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return video_tensor, torch.tensor(labels.values, dtype=torch.long)

# ------------------------------
# Model Modules
# ------------------------------
class CrossAttention(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(spatial_dim, output_dim)
        self.key   = nn.Linear(temporal_dim, output_dim)
        self.value = nn.Linear(temporal_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, spatial_feat, temporal_feat):
        query = self.query(spatial_feat).unsqueeze(1)
        key = self.key(temporal_feat).unsqueeze(2)
        attn = self.softmax(torch.bmm(query, key))
        value = self.value(temporal_feat)
        return attn.squeeze(-1) * value

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
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
        return x * spatial_attn

class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim=1536, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.proj = nn.Identity()  # Do not change the dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 500, feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    def forward(self, x):
        x = self.proj(x)
        B, T, C = x.shape
        pos_emb = self.pos_embedding[:, :T, :]
        x = x + pos_emb
        out = checkpoint(self.encoder, x, use_reentrant=False)
        return out

class TemporalBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        return torch.cat((h_n[-2], h_n[-1]), dim=1)

class TemporalTCN(nn.Module):
    def __init__(self, input_size, hidden_size=128, kernel_size=3, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size,
                                    padding=dilation*(kernel_size-1)//2, dilation=dilation))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x = x.transpose(1,2)
        out = self.network(x)
        return out.mean(dim=2)

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5, output_dim=None):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class CORNHead(nn.Module):
    def __init__(self, size_in, num_classes=4):
        super().__init__()
        try:
            self.coral = CoralLayer(size_in=size_in, num_classes=num_classes)
        except TypeError:
            self.coral = CoralLayer(size_in, num_classes)
    def forward(self, x):
        return self.coral(x)

# ------------------------------
# Loss Functions
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
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

def weighted_cross_entropy_loss(inputs, targets, weights):
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    return loss_fn(inputs, targets)

def emd_loss(inputs, targets):
    inputs = torch.softmax(inputs, dim=1)
    cum_inputs = torch.cumsum(inputs, dim=1)
    num_classes = inputs.size(1)
    target_dist = torch.zeros_like(inputs)
    for i in range(num_classes):
        target_dist[:, i] = (targets == i).float()
    cum_targets = torch.cumsum(target_dist, dim=1)
    return torch.mean(torch.abs(cum_inputs - cum_targets))

def compute_loss(loss_type, logits, targets, class_weights=None):
    if loss_type == "corn":
        losses = [corn_loss(logits[i], targets[:, i], NUM_DAISEE_CLASSES) for i in range(4)]
        return sum(losses) / 4.0
    elif loss_type == "focal":
        focal = FocalLoss()
        losses = []
        for i in range(4):
            fake_logits = torch.cat([logits[i], torch.zeros_like(logits[i][:, :1])], dim=1)
            losses.append(focal(fake_logits, targets[:, i]))
        return sum(losses) / 4.0
    elif loss_type == "weighted_ce":
        losses = []
        for i in range(4):
            fake_logits = torch.cat([logits[i], torch.zeros_like(logits[i][:, :1])], dim=1)
            losses.append(weighted_cross_entropy_loss(fake_logits, targets[:, i], class_weights))
        return sum(losses) / 4.0
    elif loss_type == "emd":
        losses = []
        for i in range(4):
            fake_logits = torch.cat([logits[i], torch.zeros_like(logits[i][:, :1])], dim=1)
            losses.append(emd_loss(fake_logits, targets[:, i]))
        return sum(losses) / 4.0
    elif loss_type == "mse":
        losses = []
        for i in range(4):
            fake_logits = torch.cat([logits[i], torch.zeros_like(logits[i][:, :1])], dim=1)
            preds = torch.softmax(fake_logits, dim=1).argmax(dim=1).float()
            one_hot = torch.nn.functional.one_hot(targets[:, i], num_classes=fake_logits.size(1)).float().to(preds.device)
            losses.append(nn.MSELoss()(torch.softmax(fake_logits, dim=1), one_hot))
        return sum(losses) / 4.0
    else:
        raise ValueError("Unknown loss type")

# ------------------------------
# Gradual Unfreezing Function
# ------------------------------
def unfreeze_backbone_stages(model, from_stage=6):
    if hasattr(model.backbone, "blocks"):
        for i, block in enumerate(model.backbone.blocks):
            if i >= from_stage:
                for param in block.parameters():
                    param.requires_grad = True
        if hasattr(model.backbone, "conv_head"):
            for param in model.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(model.backbone, "bn2"):
            for param in model.backbone.bn2.parameters():
                param.requires_grad = True

# ------------------------------
# Full Model: EfficientNetV2L with All Features (Multi-Task Learning)
# ------------------------------
class EfficientNetV2L_Full(nn.Module):
    def __init__(self, temporal_model="transformer", transformer_layers=2, transformer_heads=8,
                 bilstm_hidden=256, tcn_hidden=128, classifier_hidden=256, classifier_fc_layers=1,
                 dropout_rate=0.5, loss_type="corn", use_spatial_dropout=True, dummy_resolution=224):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.to(device)
        dummy = torch.randn(1, 3, dummy_resolution, dummy_resolution).to(device)
        with torch.no_grad():
            dummy_out = self.backbone(dummy).flatten(1)
        self.feature_dim = dummy_out.shape[1]
        print(f"Backbone feature dimension set to: {self.feature_dim}")

        self.cbam = CBAM(self.feature_dim, reduction=16, kernel_size=7)
        
        self.temporal_model_type = temporal_model
        if temporal_model == "bilstm":
            self.temporal_model = TemporalBiLSTM(self.feature_dim, hidden_size=bilstm_hidden, num_layers=1)
            temporal_out_dim = 2 * bilstm_hidden
        elif temporal_model == "tcn":
            self.temporal_model = TemporalTCN(self.feature_dim, hidden_size=tcn_hidden, kernel_size=3, num_layers=2)
            temporal_out_dim = tcn_hidden
        else:
            self.temporal_model = TemporalTransformer(feature_dim=self.feature_dim, nhead=transformer_heads,
                                                        num_layers=transformer_layers, dim_feedforward=2048, dropout=0.1)
            temporal_out_dim = self.feature_dim
        
        self.spatial_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.temporal_proj = nn.Linear(temporal_out_dim, classifier_hidden)
        self.cross_attn = CrossAttention(spatial_dim=self.feature_dim, temporal_dim=temporal_out_dim, output_dim=classifier_hidden)
        if use_spatial_dropout:
            self.dropout = SpatialDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        fusion_dim = classifier_hidden * 3
        if classifier_fc_layers > 0:
            self.classifier_head = ClassifierHead(fusion_dim, classifier_hidden, num_layers=classifier_fc_layers, dropout=dropout_rate)
            final_dim = classifier_hidden
        else:
            self.classifier_head = nn.Identity()
            final_dim = fusion_dim
        
        self.task_heads = nn.ModuleDict({
            "engagement": CORNHead(final_dim, NUM_DAISEE_CLASSES),
            "boredom":    CORNHead(final_dim, NUM_DAISEE_CLASSES),
            "confusion":  CORNHead(final_dim, NUM_DAISEE_CLASSES),
            "frustration": CORNHead(final_dim, NUM_DAISEE_CLASSES)
        })
        self.loss_type = loss_type
        
    def forward_backbone(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.amp.autocast(device_type="cuda" , enabled=False):
            out = self.backbone(x)
        return out.flatten(1)
    
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)
            feats = checkpoint(self.forward_backbone, x, use_reentrant=False)
            feats = feats.view(B, T, self.feature_dim)
        elif x.dim() == 3:
            feats = x
            B, T, _ = feats.size()
        else:
            raise ValueError("Input tensor must have 3 or 5 dimensions.")
        if feats.shape[2] != self.feature_dim:
            diff = self.feature_dim - feats.shape[2]
            if diff > 0:
                padding = torch.zeros(feats.size(0), feats.size(1), diff, device=feats.device, dtype=feats.dtype)
                feats = torch.cat([feats, padding], dim=2)
            else:
                feats = feats[:, :, :self.feature_dim]
        feats = feats.permute(0, 2, 1)
        feats = self.cbam(feats)
        feats = feats.permute(0, 2, 1)
        spatial_feature = feats.mean(dim=1)
        if self.temporal_model_type == "bilstm":
            temporal_context = self.temporal_model(feats)
        elif self.temporal_model_type == "tcn":
            temporal_context = self.temporal_model(feats)
        else:
            transformer_out = self.temporal_model(feats)
            temporal_context = transformer_out.mean(dim=1)
        sp_proj = self.spatial_proj(spatial_feature)
        tp_proj = self.temporal_proj(temporal_context)
        cross_context = self.cross_attn(spatial_feature, temporal_context)
        fusion = torch.cat((sp_proj, tp_proj, cross_context), dim=1)
        fusion = self.dropout(fusion)
        fusion = self.classifier_head(fusion)
        logits = {task: head(fusion) for task, head in self.task_heads.items()}
        return logits["engagement"], logits["boredom"], logits["confusion"], logits["frustration"]

# ------------------------------
# Optimizer Selection with Optional Lookahead
# ------------------------------
def get_optimizer(optimizer_name, params, lr, weight_decay, lookahead=False):
    optim_cls = OPTIMIZER_OPTIONS.get(optimizer_name, optim.AdamW)
    optimizer = optim_cls(params, lr=lr, weight_decay=weight_decay)
    if lookahead and "radam" in optimizer_name:
        try:
            from torch_optimizer import Lookahead
            optimizer = Lookahead(optimizer, k=LOOKAHEAD_K, alpha=LOOKAHEAD_ALPHA)
        except ImportError:
            print("Lookahead not available; proceeding without it.")
    return optimizer

def get_optimizer_state_dict(optimizer):
    # If the optimizer is a Lookahead instance, return the inner optimizer's state_dict.
    try:
        return optimizer.state_dict()
    except AttributeError as e:
        # As a fallback, try accessing the inner optimizer (if exists).
        if hasattr(optimizer, "optimizer"):
            return optimizer.optimizer.state_dict()
        else:
            raise e

# ------------------------------
# Training Function with Progressive Resolution & Adaptive Grad Accumulation
# ------------------------------
def progressive_train_model_full(model, total_epochs, lr, checkpoint_path, batch_size,
                                 loss_type, optimizer_name="adamw", lookahead=False,
                                 patience=5, grad_accum_steps=GRADIENT_ACCUM_STEPS):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    for p in backbone_params:
        p.requires_grad = False
    for p in other_params:
        p.requires_grad = True
    optimizer = get_optimizer(optimizer_name, backbone_params + other_params, lr, weight_decay=1e-4, lookahead=lookahead)
    def lr_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler()
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    current_epoch = 0
    train_csv_local = train_csv
    val_csv_local = val_csv
    accum_steps = grad_accum_steps
    for schedule_index, (res, ep) in enumerate(PROG_SCHEDULE):
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv_local, num_frames=NUM_FRAMES, resolution=res,
                                          transform=transform, lmdb_map_size=1 * 1024**3)
        val_lmdb = convert_pkl_to_lmdb(val_csv_local, num_frames=NUM_FRAMES, resolution=res,
                                        transform=transform, lmdb_map_size=1 * 1024**3)
        train_set = VideoDatasetLMDB(train_csv_local, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set = VideoDatasetLMDB(val_csv_local, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
        all_labels = train_set.data["Engagement"].astype(int).values
        class_counts = np.bincount(all_labels)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
        sampler = WeightedRandomSampler([class_weights[label].item() for label in all_labels],
                                        num_samples=len(all_labels), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        if schedule_index == 1:
            unfreeze_backbone_stages(model, from_stage=6)
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
            optimizer = get_optimizer(optimizer_name, backbone_params + other_params, lr, weight_decay=1e-4, lookahead=lookahead)
            print("Gradual unfreezing: last blocks of the backbone are now unfrozen.")
        for epoch_i in range(ep):
            epoch_global = current_epoch + 1
            print(f"Progressive Training: Epoch {epoch_global}/{total_epochs} at resolution {res}x{res}")
            model.train()
            running_loss = 0.0
            total_samples = 0
            batch_count = 0
            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(features)
                    loss = compute_loss(loss_type, logits, labels, class_weights)
                scaler.scale(loss / accum_steps).backward()
                batch_count += 1
                if batch_count % accum_steps == 0:
                    scaler.step(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                del features, labels, loss
                if (i + 1) % 30 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            train_loss = running_loss / total_samples
            scheduler.step()
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(features)
                        loss = compute_loss(loss_type, logits, labels, class_weights)
                    val_loss_sum += loss.item() * features.size(0)
                    val_count += features.size(0)
            val_loss = val_loss_sum / val_count
            print(f"Epoch {epoch_global}/{total_epochs} at {res}x{res} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                state = {
                    "epoch": epoch_global,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": get_optimizer_state_dict(optimizer)
                }
                temp_path = checkpoint_path.with_suffix(".tmp")
                torch.save(state, temp_path, _use_new_zipfile_serialization=False)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                temp_path.rename(checkpoint_path)
            else:
                early_stop_counter += 1
                if early_stop_counter % 2 == 0:
                    accum_steps *= 2
                    print(f"Adaptive Grad Accumulation: Increasing accumulation steps to {accum_steps}")
            current_epoch += 1
            if current_epoch >= total_epochs:
                break
    return best_val_loss

# ------------------------------
# Evaluation Function with TTA
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, test_loader, tta=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if tta:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits1 = model(frames)
                frames_flipped = torch.flip(frames, dims=[3])
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits2 = model(frames_flipped)
                logits = tuple((l1 + l2) / 2.0 for l1, l2 in zip(logits1, logits2))
            else:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(frames)
            def corn_logits_to_label(logits):
                sig = torch.sigmoid(logits)
                return torch.sum(sig > 0.5, dim=1)
            pred_eng = corn_logits_to_label(logits[0])
            pred_bor = corn_logits_to_label(logits[1])
            pred_con = corn_logits_to_label(logits[2])
            pred_fru = corn_logits_to_label(logits[3])
            stacked_preds = torch.stack([pred_eng, pred_bor, pred_con, pred_fru], dim=1)
            all_preds.append(stacked_preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metrics_names = ["Engagement", "Boredom", "Confusion", "Frustration"]
    for i, metric in enumerate(metrics_names):
        print(f"Classification report for {metric}:")
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {metric}")
        plt.colorbar()
        plt.xticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
        plt.yticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

# ------------------------------
# Hyperparameter Tuning with Optuna (.db file)
# ------------------------------
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')  
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)  
    
    db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_corn_cbam__full.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        print(f"Database created/connected at: {db_path}")
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

    def objective(trial):
        torch.cuda.empty_cache()
        gc.collect()
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        temporal_model_choice = trial.suggest_categorical("temporal_model", ["bilstm", "transformer", "tcn"])
        transformer_layers = trial.suggest_int("transformer_layers", 2, 3) if temporal_model_choice == "transformer" else 2
        # Restrict transformer_heads to only 8 to guarantee divisibility.
        transformer_heads = trial.suggest_categorical("transformer_heads", [8]) if temporal_model_choice == "transformer" else 8
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
        classifier_hidden = trial.suggest_categorical("classifier_hidden", [256, 512])
        classifier_fc_layers = trial.suggest_int("classifier_fc_layers", 0, 2)
        loss_choice = trial.suggest_categorical("loss_type", ["corn", "focal", "weighted_ce", "emd", "mse"])
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "radam"])
        lookahead = trial.suggest_categorical("lookahead", [True, False])
        
        model = EfficientNetV2L_Full(temporal_model=temporal_model_choice,
                                      transformer_layers=transformer_layers,
                                      transformer_heads=transformer_heads,
                                      bilstm_hidden=256,
                                      tcn_hidden=128,
                                      classifier_hidden=classifier_hidden,
                                      classifier_fc_layers=classifier_fc_layers,
                                      dropout_rate=dropout_rate,
                                      loss_type=loss_choice).to(device)
        
        trial_checkpoint = MODEL_DIR / f"trial_checkpoint_corn_cbam_{trial.number}.pth"
        trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        total_epochs = sum(e for _, e in PROG_SCHEDULE)
        best_loss = progressive_train_model_full(model,
                                                 total_epochs=total_epochs,
                                                 lr=lr,
                                                 checkpoint_path=trial_checkpoint,
                                                 batch_size=batch_size,
                                                 loss_type=loss_choice,
                                                 optimizer_name=optimizer_name,
                                                 lookahead=lookahead,
                                                 patience=3)
        return best_loss

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5),
                                study_name="efficientnetv2l_full_study",
                                storage=f"sqlite:///{db_path}",
                                load_if_exists=True)

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
    # Final Training using Enhanced Two-Phase Strategy
    # ------------------------------

    # Reload the best trial from the saved study to extract hyperparameters.
    study = optuna.load_study(study_name="efficientnetv2l_full_study", storage=f"sqlite:///{db_path}")
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
    best_trial = min(valid_trials, key=lambda t: t.value)
    params = best_trial.params
    print(f"Best trial parameters: {params}")

    # Handle batch size and gradient accumulation for both phases
    final_effective_batch_size = params.get("batch_size", 8)  # From hyperparameter tuning
    per_iteration_batch_size = min(final_effective_batch_size, 4)  # Fit in 8GB VRAM
    grad_accum_steps = final_effective_batch_size // per_iteration_batch_size
    lr = params.get("lr", 1e-4)

    final_model = EfficientNetV2L_Full(
        temporal_model=params.get("temporal_model", "bilstm"),
        transformer_layers=params.get("transformer_layers", 2),
        transformer_heads=params.get("transformer_heads", 8),
        classifier_hidden=params.get("classifier_hidden", 256),
        classifier_fc_layers=params.get("classifier_fc_layers", 1),
        dropout_rate=params.get("dropout_rate", 0.5),
        loss_type=params.get("loss_type", "corn")
    ).to(device)

    final_checkpoint = MODEL_DIR / "final_model_eff_v2l_full_corn_cbam_checkpoint.pth"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if LMDB phase is already complete before starting training
    lmdb_complete_checkpoint = MODEL_DIR / "lmdb_phase_complete.pth"
    skip_to_phase2 = False
    difficult_examples = []
    difficult_examples_file = CACHE_DIR / "difficult_examples.pkl"
    scaler = GradScaler()  # Initialize scaler here for use in both paths

    if lmdb_complete_checkpoint.exists():
        print(f"\n=== Found LMDB phase completion checkpoint, checking if valid ===")
        try:
            state = torch.load(lmdb_complete_checkpoint, map_location=device, weights_only=False)
            if "lmdb_phase_complete" in state and state["lmdb_phase_complete"]:
                print("LMDB Phase already completed. Loading checkpoint and skipping to Phase 2.")
                final_model.load_state_dict(state["model_state_dict"])
                optimizer = get_optimizer(params.get("optimizer", "adamw"), final_model.parameters(), 
                                        lr, weight_decay=1e-4, lookahead=params.get("lookahead", False))
                optimizer.load_state_dict(state["optimizer_state_dict"])
                best_val_loss = state["best_val_loss"]
                if "difficult_examples" in state:
                    difficult_examples = state["difficult_examples"]
                skip_to_phase2 = True
            else:
                print("Checkpoint found but doesn't contain LMDB phase completion flag.")
        except Exception as e:
            print(f"Error loading LMDB checkpoint: {e}")
            print("Starting from Phase 1")

    if not skip_to_phase2:
        # ------------------------------
        # Phase 1: Optimized LMDB Training
        # ------------------------------
        print("\n=== Phase 1: Optimized LMDB Training ===")

        # Focus on higher resolutions for efficiency and quality but keep 30 epochs total
        lmdb_schedule = [(224, 10), (300, 20)]  # Match the 30 epochs from hyperparameter tuning
        total_epochs = sum(ep for _, ep in lmdb_schedule)

        # Set up optimizer with cosine LR scheduler and warmup
        optimizer = get_optimizer(params.get("optimizer", "adamw"), final_model.parameters(), 
                                lr, weight_decay=1e-4, lookahead=params.get("lookahead", False))

        def lr_lambda(epoch):
            warmup_epochs = 1
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        best_val_loss = float('inf')
        current_epoch = 0

        # Check for periodic checkpoints from previous run
        periodic_checkpoints = sorted(list(MODEL_DIR.glob("phase1_checkpoint_epoch_*.pth")))
        if periodic_checkpoints:
            latest_checkpoint = periodic_checkpoints[-1]
            print(f"Found periodic checkpoint: {latest_checkpoint}")
            try:
                state = torch.load(latest_checkpoint, map_location=device, weights_only=False)
                final_model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                current_epoch = state["current_epoch"]
                best_val_loss = state.get("best_val_loss", float('inf'))
                
                # Find which schedule index and epochs we should start from
                if "res" in state and "schedule_index" in state:
                    res_to_find = state["res"]
                    start_schedule_index = state["schedule_index"]
                    
                    # Skip completed resolutions
                    for i, (res, _) in enumerate(lmdb_schedule):
                        if i < start_schedule_index:
                            # Skip this resolution entirely
                            current_epoch += lmdb_schedule[i][1]
                            continue
                        elif i == start_schedule_index:
                            # Resume from this resolution
                            epochs_done = state.get("epochs_at_res", 0)
                            print(f"Resuming training at resolution {res}x{res} from epoch {epochs_done+1}")
                            lmdb_schedule[i] = (res, lmdb_schedule[i][1] - epochs_done)
                            break
                    
                print(f"Resuming from overall epoch {current_epoch}")
            except Exception as e:
                print(f"Error loading periodic checkpoint: {e}")
                print("Starting fresh training")
                current_epoch = 0
        else:
            print("No periodic checkpoints found. Starting fresh training.")

        for schedule_index, (res, epochs) in enumerate(lmdb_schedule):
            print(f"\nTraining at resolution {res}x{res} for {epochs} epochs")
            
            # Prepare LMDB datasets
            train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res, 
                                            transform=get_transform(res))
            val_lmdb = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res, 
                                        transform=get_transform(res))
            
            train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
            val_set = VideoDatasetLMDB(val_csv, val_lmdb, num_frames=NUM_FRAMES, resolution=res)
            
            # Class weights to handle imbalance
            all_labels = train_set.data["Engagement"].astype(int).values
            class_counts = np.bincount(all_labels)
            class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
            
            train_loader = DataLoader(train_set, batch_size=per_iteration_batch_size, shuffle=True, 
                                    num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=per_iteration_batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)
            
            # Unfreeze backbone at higher resolution
            if res >= 224:
                unfreeze_backbone_stages(final_model, from_stage=6)
                print("Unfreezing last blocks of backbone")
            
            for epoch in range(epochs):
                current_epoch += 1
                print(f"LMDB Epoch {current_epoch}/{total_epochs} | Resolution: {res}x{res}")
                
                # Training loop
                final_model.train()
                running_loss = 0.0
                total_samples = 0
                optimizer.zero_grad()
                
                for i, (features, labels) in enumerate(tqdm(train_loader, desc="LMDB Training")):
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = final_model(features)
                        loss = compute_loss(params.get("loss_type", "corn"), logits, labels, class_weights)
                        
                    scaler.scale(loss / grad_accum_steps).backward()
                    
                    if (i + 1) % grad_accum_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                    running_loss += loss.item() * features.size(0)
                    total_samples += features.size(0)
                    
                    # Clean up GPU memory periodically
                    if (i + 1) % 20 == 0:
                        torch.cuda.empty_cache()
                        
                train_loss = running_loss / total_samples
                scheduler.step()
                
                # Validation loop
                final_model.eval()
                val_loss_sum = 0.0
                val_count = 0
                
                # Only collect difficult examples in final epoch at highest resolution
                collect_difficult = (epoch == epochs - 1 and res == 300)
                epoch_difficult_examples = []  # Collect in a separate list first
                
                with torch.no_grad():
                    for idx, (features, labels) in enumerate(val_loader):
                        features = features.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        with autocast(device_type='cuda', dtype=torch.float16):
                            logits = final_model(features)
                            loss = compute_loss(params.get("loss_type", "corn"), logits, labels, class_weights)
                            
                        val_loss_sum += loss.item() * features.size(0)
                        val_count += features.size(0)
                        
                        # Identify difficult examples in last epoch - with explicit error handling
                        if collect_difficult:
                            try:
                                # Check task predictions
                                batch_errors = []
                                for i in range(4):
                                    task_logits = logits[i]
                                    task_labels = labels[:, i]
                                    
                                    # Get predictions from CORN outputs
                                    sig_logits = torch.sigmoid(task_logits)
                                    pred_labels = torch.sum(sig_logits > 0.5, dim=1)
                                    
                                    # Find errors
                                    errors = (pred_labels != task_labels)
                                    batch_errors.append(errors)
                                
                                # Combine errors across tasks
                                combined_errors = torch.stack(batch_errors).sum(dim=0)
                                error_indices = torch.nonzero(combined_errors).squeeze().cpu().numpy()
                                
                                # Handle scalar case
                                if isinstance(error_indices, np.ndarray):
                                    if len(error_indices.shape) == 0 and error_indices.size > 0:
                                        error_indices = np.array([error_indices])
                                elif isinstance(error_indices, np.int64):  # Handle scalar value
                                    error_indices = np.array([error_indices])
                                
                                if isinstance(error_indices, np.ndarray) and error_indices.size > 0:
                                    base_idx = idx * per_iteration_batch_size
                                    for i in error_indices:
                                        if base_idx + i < len(val_set.valid_indices):
                                            epoch_difficult_examples.append(val_set.valid_indices[base_idx + i])
                            except Exception as e:
                                print(f"Error collecting difficult examples: {e}")
                                # Continue without stopping the training
                        
                val_loss = val_loss_sum / val_count
                print(f"LMDB Epoch {current_epoch}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Save checkpoint if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state = {
                        "epoch": current_epoch,
                        "best_val_loss": best_val_loss,
                        "model_state_dict": final_model.state_dict(),
                        "optimizer_state_dict": get_optimizer_state_dict(optimizer)
                    }
                    temp_path = final_checkpoint.with_suffix(".tmp")
                    torch.save(state, temp_path, _use_new_zipfile_serialization=False)
                    if final_checkpoint.exists():
                        final_checkpoint.unlink()
                    temp_path.rename(final_checkpoint)
                
                # Save difficult examples at the end of each epoch in the collection phase
                if collect_difficult and epoch_difficult_examples:
                    difficult_examples.extend(epoch_difficult_examples)
                    # Save immediately after collection to prevent loss
                    try:
                        with open(difficult_examples_file, "wb") as f:
                            pickle.dump(difficult_examples, f)
                        print(f"Saved {len(difficult_examples)} difficult examples to {difficult_examples_file}")
                    except Exception as e:
                        print(f"Error saving difficult examples: {e}")
                
                # Save periodic checkpoint
                if (current_epoch) % 5 == 0 or epoch == epochs-1:  # Save every 5 epochs and at the end of each resolution
                    periodic_state = {
                        "model_state_dict": final_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "current_epoch": current_epoch,
                        "best_val_loss": best_val_loss,
                        "res": res,
                        "schedule_index": schedule_index,
                        "epochs_at_res": epoch + 1
                    }
                    periodic_path = MODEL_DIR / f"phase1_checkpoint_epoch_{current_epoch}.pth"
                    torch.save(periodic_state, periodic_path, _use_new_zipfile_serialization=False)
                    print(f"Saved periodic checkpoint at epoch {current_epoch}")
        
        # Make sure to close any LMDB environments properly
        if hasattr(train_set, 'env') and train_set.env is not None:
            train_set.env.close()
        if hasattr(val_set, 'env') and val_set.env is not None:
            val_set.env.close()
            
        # Clean up to free memory before Phase 2
        del train_loader, val_loader, train_set, val_set
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n--- LMDB Training Phase Complete ---")
        
        # Save LMDB phase completion checkpoint
        torch.save({
            "model_state_dict": final_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "difficult_examples": difficult_examples,
            "lmdb_phase_complete": True
        }, lmdb_complete_checkpoint, _use_new_zipfile_serialization=False)
        print(f"Saved LMDB phase completion checkpoint to {lmdb_complete_checkpoint}")
        
    # ------------------------------
    # Phase 2: Targeted Raw-Image Fine-Tuning
    # ------------------------------
    print("\n=== Phase 2: Targeted Raw-Image Fine-Tuning ===")

    # Load best model if we haven't already skipped to phase 2
    if not skip_to_phase2:
        # Load from LMDB phase completion checkpoint (which we just saved)
        try:
            state = torch.load(lmdb_complete_checkpoint, map_location=device, weights_only=False)
            final_model.load_state_dict(state["model_state_dict"])
            optimizer = get_optimizer(params.get("optimizer", "adamw"), final_model.parameters(), 
                                    lr, weight_decay=1e-4, lookahead=params.get("lookahead", False))
            optimizer.load_state_dict(state["optimizer_state_dict"])
            best_val_loss = state["best_val_loss"]
            if "difficult_examples" in state:
                difficult_examples = state["difficult_examples"]
        except Exception as e:
            print(f"Error loading LMDB checkpoint: {e}")
            print("Falling back to best model checkpoint")
            state = torch.load(final_checkpoint, map_location=device, weights_only=False)
            final_model.load_state_dict(state["model_state_dict"])

    final_model.to(device)

    # Load difficult examples if we don't already have them
    if len(difficult_examples) == 0 and difficult_examples_file.exists():
        try:
            with open(difficult_examples_file, "rb") as f:
                difficult_examples = pickle.load(f)
            print(f"Loaded {len(difficult_examples)} difficult examples from file")
        except Exception as e:
            print(f"Error loading difficult examples: {e}")

    # Setup for raw image fine-tuning
    print(f"Found {len(difficult_examples)} difficult examples for targeted fine-tuning")

    # Limit difficult examples for feasibility - but using 2000 for best results
    if len(difficult_examples) > 2000:
        print(f"Limiting to 600 difficult examples (from {len(difficult_examples)})")
        difficult_examples = difficult_examples[:2000]

    # Create validation dataset with raw images
    val_ft_transform = get_transform(300)  # Always use highest resolution
    val_ft_set = VideoDatasetRaw(val_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=val_ft_transform)

    # Create targeted training dataset with difficult examples
    targeted_train_set = TargetedVideoDataset(
        train_csv, FRAMES_DIR, difficult_examples, 
        num_frames=NUM_FRAMES, transform=val_ft_transform
    )

    # Use smaller batch size for raw images due to memory constraints
    targeted_batch_size = 2
    targeted_grad_accum = final_effective_batch_size // targeted_batch_size

    targeted_train_loader = DataLoader(
        targeted_train_set, batch_size=targeted_batch_size,
        shuffle=True, num_workers=0, pin_memory=True  # Set num_workers=0 to avoid multiprocessing issues
    )

    val_ft_loader = DataLoader(
        val_ft_set, batch_size=targeted_batch_size,
        shuffle=False, num_workers=0, pin_memory=True  # Set num_workers=0 to avoid multiprocessing issues
    )

    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 0.1

    # Fine-tuning with improved setup - more epochs and early stopping
    ft_epochs = 8  # Increased from 3 to 8
    patience = 3   # Stop if no improvement for 3 consecutive epochs
    best_ft_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(ft_epochs):
        # Decay learning rate slightly each epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1 * (0.9 ** epoch)
        
        print(f"Fine-tuning epoch {epoch+1}/{ft_epochs} (lr: {optimizer.param_groups[0]['lr']:.6f})")
        
        final_model.train()
        running_loss = 0.0
        total_samples = 0
        optimizer.zero_grad()
        
        for i, (features, labels) in enumerate(tqdm(targeted_train_loader, desc="Raw Fine-tuning")):
            features, labels = features.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = final_model(features)
                loss = compute_loss(final_model.loss_type, outputs, labels)
            scaler.scale(loss).backward()
            if (i + 1) % targeted_grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
        
        if total_samples > 0:
            train_loss = running_loss / total_samples
        else:
            train_loss = float('inf')
        
        # Validation
        final_model.eval()
        val_loss_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_ft_loader, desc="Validation"):
                features, labels = features.to(device), labels.to(device)
                outputs = final_model(features)
                loss = compute_loss(final_model.loss_type, outputs, labels)
                val_loss_sum += loss.item() * features.size(0)
                val_count += features.size(0)
        
        if val_count > 0:
            val_loss = val_loss_sum / val_count
        else:
            val_loss = float('inf')
            
        print(f"FT Epoch {epoch+1}/{ft_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if improved and reset patience counter
        if val_loss < best_ft_val_loss:
            best_ft_val_loss = val_loss
            epochs_no_improve = 0
            print(f"Saving improved model with val_loss: {val_loss:.4f}")
            torch.save({
                "model_state_dict": final_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": min(best_val_loss, val_loss),  # Keep track of the global best val loss
                "ft_epoch": epoch + 1,
            }, final_checkpoint, _use_new_zipfile_serialization=False)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("\n--- Targeted Raw-Image Fine-Tuning Complete ---")

    # Load best model for final evaluation
    state = torch.load(final_checkpoint, map_location=device, weights_only=False)
    final_model.load_state_dict(state["model_state_dict"])
    final_model.to(device)

    # ------------------------------
    # Enhanced Evaluation with TTA
    # ------------------------------
    def evaluate_model_with_enhanced_tta(model, test_loader):
        """Enhanced evaluation with test-time augmentation"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Evaluating"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                # Get predictions from CORN outputs
                preds = []
                for i in range(4):
                    sig_logits = torch.sigmoid(outputs[i])
                    pred_labels = torch.sum(sig_logits > 0.5, dim=1)
                    preds.append(pred_labels)
                
                all_preds.append(torch.stack(preds, dim=1).cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics_names = ["Engagement", "Boredom", "Confusion", "Frustration"]
        for i, metric in enumerate(metrics_names):
            print(f"Classification report for {metric}:")
            print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
            cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {metric}")
            plt.colorbar()
            plt.xticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
            plt.yticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(f"{metric}_confusion_matrix.png")
            plt.close()

    # Prepare test data and evaluate
    print("\n=== Final Evaluation with Enhanced TTA ===")
    test_transform = get_transform(300)  # Highest resolution for evaluation
    test_set = VideoDatasetRaw(test_csv, FRAMES_DIR, num_frames=NUM_FRAMES, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=per_iteration_batch_size//2, shuffle=False, num_workers=0, pin_memory=True)

    # Run enhanced evaluation
    evaluate_model_with_enhanced_tta(final_model, test_loader)

    torch.cuda.empty_cache()
    gc.collect()
    print("\n--- Evaluation Complete ---")