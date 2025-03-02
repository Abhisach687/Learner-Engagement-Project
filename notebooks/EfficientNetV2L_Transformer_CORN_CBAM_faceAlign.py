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
from torch.utils.data.sampler import WeightedRandomSampler  # for class imbalance
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import timm  # pip install timm

# --- Additional libraries for new features ---
from torch.utils.checkpoint import checkpoint
from facenet_pytorch import MTCNN  # pip install facenet-pytorch

# Try to import coral_pytorch; if unavailable, use minimal fallback implementations.
try:
    from coral_pytorch.layer import CoralLayer
    from coral_pytorch.losses import corn_loss
except ImportError:
    logging.warning("coral_pytorch not found; using fallback implementations for CoralLayer and corn_loss.")

    class CoralLayer(nn.Module):
        def __init__(self, num_features, num_classes):
            super().__init__()
            self.linear = nn.Linear(num_features, num_classes - 1)
        def forward(self, x):
            return self.linear(x)

    def corn_loss(logits, targets):
        bce = nn.BCEWithLogitsLoss()
        num_classes = logits.size(1) + 1
        target_matrix = torch.zeros_like(logits, dtype=torch.float)
        for k in range(1, num_classes):
            target_matrix[:, k - 1] = (targets >= k).float()
        return bce(logits, target_matrix)

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

# Enable face alignment in the preprocessing pipeline
FACE_ALIGNMENT = True

# Number of ordinal classes per DAiSEE dimension: [0,1,2,3]
NUM_DAISEE_CLASSES = 4

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
# MTCNN for face alignment (✓)
# ------------------------------
mtcnn_detector = MTCNN(image_size=300, margin=20, post_process=True, device=device)

def align_and_crop_face(img: Image.Image, target_size: int = 300):
    """
    Use MTCNN to detect and align a face in 'img'.
    Returns a cropped-aligned face PIL Image of size (target_size, target_size),
    or None if detection fails.
    """
    face_tensor = mtcnn_detector(img)
    if face_tensor is None:
        return None
    face_pil = transforms.ToPILImage()(face_tensor).convert('RGB')
    return face_pil

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
    tag = "_facealign" if FACE_ALIGNMENT else ""
    lmdb_path = CACHE_DIR / f"lmdb_{csv_file.stem}_frame_{num_frames}_{resolution}{tag}"
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

    def forward_backbone(batch_imgs):
        return backbone(batch_imgs)

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
                if FACE_ALIGNMENT:
                    aligned_face = align_and_crop_face(img, target_size=resolution)
                    if aligned_face is not None:
                        img = aligned_face
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    feat = checkpoint(forward_backbone, tensor)
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
    def __init__(self, csv_file, lmdb_path, num_frames=50, resolution=224):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        self.resolution = resolution
        pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
        with open(pkl_file, "rb") as f:
            cache = pickle.load(f)
        self.valid_indices = cache["valid_indices"]
        self.file_paths = cache["precomputed_frames"]
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
        original_idx = self.data.index[idx]
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
# CORN: Ordinal Regression Setup (✓)
# ------------------------------
class CORNHead(nn.Module):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.coral = CoralLayer(num_features=in_features, num_classes=num_classes)
    def forward(self, x):
        return self.coral(x)

def compute_corn_loss(logits, targets):
    return corn_loss(logits, targets)

# ------------------------------
# CBAM & CrossAttention Modules (✓)
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
# Transformer-based Temporal Encoder (✓)
# ------------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim=1536, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
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
        B, T, C = x.shape
        pos_emb = self.pos_embedding[:, :T, :]
        x = x + pos_emb
        out = checkpoint(self.encoder, x)
        return out

# ------------------------------
# Full Model (✓)
# ------------------------------
class EfficientNetV2L_Transformer_CORN_CBAM(nn.Module):
    def __init__(self, transformer_layers=2, transformer_heads=8, dropout_rate=0.5, classifier_hidden=256):
        super(EfficientNetV2L_Transformer_CORN_CBAM, self).__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = 1280
        self.cbam = CBAM(self.feature_dim, reduction=16, kernel_size=7)
        self.transformer = TemporalTransformer(
            feature_dim=self.feature_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.cross_attn = CrossAttention(self.feature_dim)
        self.temporal_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.spatial_proj = nn.Linear(self.feature_dim, classifier_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.corn_head_engagement = CORNHead(classifier_hidden * 2, NUM_DAISEE_CLASSES)
        self.corn_head_boredom    = CORNHead(classifier_hidden * 2, NUM_DAISEE_CLASSES)
        self.corn_head_confusion  = CORNHead(classifier_hidden * 2, NUM_DAISEE_CLASSES)
        self.corn_head_frustr     = CORNHead(classifier_hidden * 2, NUM_DAISEE_CLASSES)
    def forward_backbone(self, x):
        return self.backbone(x)
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)
            feats = checkpoint(self.forward_backbone, x)
            feats = feats.view(B, T, self.feature_dim)
        elif x.dim() == 3:
            feats = x
            B, T, _ = feats.size()
        else:
            raise ValueError("Input tensor must have shape (B, T, C, H, W) or (B, T, feature_dim).")
        feats = feats.permute(0, 2, 1)
        feats = self.cbam(feats)
        feats = feats.permute(0, 2, 1)
        transformer_out = self.transformer(feats)
        temporal_context = transformer_out.mean(dim=1)
        spatial_feature = feats.mean(dim=1)
        cross_context = self.cross_attn(spatial_feature, temporal_context)
        sp_proj = self.spatial_proj(spatial_feature)
        cc_proj = self.temporal_proj(cross_context)
        fusion = torch.cat((sp_proj, cc_proj), dim=1)
        fusion = self.dropout(fusion)
        logits_eng = self.corn_head_engagement(fusion)
        logits_bor = self.corn_head_boredom(fusion)
        logits_con = self.corn_head_confusion(fusion)
        logits_fru = self.corn_head_frustr(fusion)
        return logits_eng, logits_bor, logits_con, logits_fru

# ------------------------------
# Gradual Unfreezing (✓)
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
# Training Function (Progressive + CORN) (✓)
# ------------------------------
def progressive_train_model(model, total_epochs, lr, checkpoint_path, batch_size,
                            patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
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

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": other_params, "lr": lr}
    ], weight_decay=1e-4)
    scaler = GradScaler()
    best_val_loss = float('inf')
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_epoch = 0
    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv   = LABELS_DIR / "ValidationLabels.csv"

    for schedule_index, (res, ep) in enumerate(PROG_SCHEDULE):
        transform = get_transform(res)
        train_lmdb = convert_pkl_to_lmdb(train_csv, num_frames=NUM_FRAMES, resolution=res,
                                         transform=transform, lmdb_map_size=1 * 1024**3)
        val_lmdb   = convert_pkl_to_lmdb(val_csv, num_frames=NUM_FRAMES, resolution=res,
                                         transform=transform, lmdb_map_size=1 * 1024**3)
        train_set = VideoDatasetLMDB(train_csv, train_lmdb, num_frames=NUM_FRAMES, resolution=res)
        val_set   = VideoDatasetLMDB(val_csv,   val_lmdb,   num_frames=NUM_FRAMES, resolution=res)

        all_labels = train_set.data["Engagement"].astype(int).values
        class_counts = np.bincount(all_labels)
        weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [weights[label] for label in all_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)

        if schedule_index == 1:
            unfreeze_backbone_stages(model, from_stage=6)
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": other_params,    "lr": lr}
            ], weight_decay=1e-4)
            print("Gradual unfreezing: last blocks of the backbone are now unfrozen.")

        for epoch_i in range(ep):
            epoch_global = current_epoch + 1
            print(f"Progressive Training: Epoch {epoch_global}/{total_epochs} at resolution {res}x{res}")
            model.train()
            running_loss = 0.0
            total_samples = 0

            for i, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
                features = features.to(device, non_blocking=True)
                labels   = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits_eng, logits_bor, logits_con, logits_fru = model(features)
                    loss_eng = compute_corn_loss(logits_eng, labels[:, 0])
                    loss_bor = compute_corn_loss(logits_bor, labels[:, 1])
                    loss_con = compute_corn_loss(logits_con, labels[:, 2])
                    loss_fru = compute_corn_loss(logits_fru, labels[:, 3])
                    loss = (loss_eng + loss_bor + loss_con + loss_fru) / 4.0
                scaler.scale(loss / gradient_accum_steps).backward()
                if (i + 1) % gradient_accum_steps == 0:
                    scaler.step(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.update()
                    optimizer.zero_grad()
                batch_sz = features.size(0)
                running_loss += loss.item() * batch_sz
                total_samples += batch_sz
                del features, labels, logits_eng, logits_bor, logits_con, logits_fru, loss
                if (i + 1) % 30 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            train_loss = running_loss / total_samples

            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device, non_blocking=True)
                    labels   = labels.to(device, non_blocking=True)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits_eng, logits_bor, logits_con, logits_fru = model(features)
                        loss_eng = compute_corn_loss(logits_eng, labels[:, 0])
                        loss_bor = compute_corn_loss(logits_bor, labels[:, 1])
                        loss_con = compute_corn_loss(logits_con, labels[:, 2])
                        loss_fru = compute_corn_loss(logits_fru, labels[:, 3])
                        loss = (loss_eng + loss_bor + loss_con + loss_fru) / 4.0
                    val_loss_sum += loss.item() * features.size(0)
                    val_count += features.size(0)
            val_loss = val_loss_sum / val_count
            print(f"Epoch {epoch_global}/{total_epochs} at {res}x{res} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state = {
                    "epoch": epoch_global,
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
                print(f"Early stopping at epoch {epoch_global}. Best val loss: {best_val_loss:.4f}")
                return best_val_loss
            current_epoch += 1
            if current_epoch >= total_epochs:
                break
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
            logits_eng, logits_bor, logits_con, logits_fru = model(frames)
            def corn_logits_to_label(logits):
                sig = torch.sigmoid(logits)
                return torch.sum(sig > 0.5, dim=1)
            pred_eng = corn_logits_to_label(logits_eng)
            pred_bor = corn_logits_to_label(logits_bor)
            pred_con = corn_logits_to_label(logits_con)
            pred_fru = corn_logits_to_label(logits_fru)
            stacked_preds = torch.stack([pred_eng, pred_bor, pred_con, pred_fru], dim=1)
            all_preds.append(stacked_preds.cpu())
            all_labels.append(labels.cpu())
    all_preds  = torch.cat(all_preds,  dim=0).numpy()
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
# Optuna Hyperparameter Tuning with Integrated Loop
# ------------------------------
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    transformer_layers = trial.suggest_int("transformer_layers", 2, 3)
    transformer_heads = trial.suggest_categorical("transformer_heads", [8, 12])
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
    classifier_hidden = trial.suggest_categorical("classifier_hidden", [256, 512])
    
    # Build the model with trial hyperparameters
    model = EfficientNetV2L_Transformer_CORN_CBAM(
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout_rate=dropout_rate,
        classifier_hidden=classifier_hidden
    ).to(device)
    
    # Use a trial-specific checkpoint file
    trial_checkpoint = MODEL_DIR / f"trial_checkpoint_{trial.number}.pth"
    total_epochs = sum(e for _, e in PROG_SCHEDULE)
    
    best_loss = progressive_train_model(
        model,
        total_epochs=total_epochs,
        lr=lr,
        checkpoint_path=trial_checkpoint,
        batch_size=batch_size,
        patience=3  # lower patience for tuning speed
    )
    return best_loss

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv   = LABELS_DIR / "ValidationLabels.csv"
    test_csv  = LABELS_DIR / "TestLabels.csv"

    # Precompute caches + LMDB for each resolution
    resolutions = [112, 224, 300]
    for csv_file in [train_csv, val_csv, test_csv]:
        for r in resolutions:
            precompute_best_frames(csv_file, FRAMES_DIR, num_frames=NUM_FRAMES, resolution=r)
            convert_pkl_to_lmdb(csv_file, num_frames=NUM_FRAMES, resolution=r,
                                transform=get_transform(r), lmdb_map_size=1 * 1024**3)

    # ------------------------------
    # Hyperparameter Tuning with Optuna (Integrated Loop)
    # ------------------------------
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5))
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
    # Final Training using Best Hyperparameters
    # ------------------------------
    total_epochs = sum(e for _, e in PROG_SCHEDULE)
    final_checkpoint = MODEL_DIR / "final_model_eff_v2l__transformer_corn_cbam_checkpoint.pth"
    model = EfficientNetV2L_Transformer_CORN_CBAM(
        transformer_layers=best_trial.params.get("transformer_layers", 2),
        transformer_heads=best_trial.params.get("transformer_heads", 8),
        dropout_rate=best_trial.params.get("dropout_rate", 0.5),
        classifier_hidden=best_trial.params.get("classifier_hidden", 256)
    ).to(device)

    if not final_checkpoint.exists():
        best_loss = progressive_train_model(
            model,
            total_epochs=total_epochs,
            lr=best_trial.params.get("lr", 1e-4),
            checkpoint_path=final_checkpoint,
            batch_size=best_trial.params.get("batch_size", 4),
            patience=5
        )
        print("Final best validation loss:", best_loss)
    else:
        print("Skipping training, final checkpoint already exists. Loading model...")
    state = torch.load(final_checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    test_transform = get_transform(300)
    tag = "_facealign" if FACE_ALIGNMENT else ""
    test_lmdb = CACHE_DIR / f"lmdb_{test_csv.stem}_frame_{NUM_FRAMES}_300{tag}"
    test_set = VideoDatasetLMDB(test_csv, test_lmdb, num_frames=NUM_FRAMES, resolution=300)
    test_loader = DataLoader(test_set, batch_size=best_trial.params.get("batch_size", 4),
                             shuffle=False, num_workers=2, pin_memory=True)

    evaluate_model(model, test_loader)
    print("\n--- Done ---")
