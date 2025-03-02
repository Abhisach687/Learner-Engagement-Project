import os, gc, math, pickle, sqlite3, datetime, random, runpy
from pathlib import Path

# Disable oneDNN custom ops for TF to avoid floating‐point round‐off warnings.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import lmdb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ------------------------------
# GLOBAL CONFIG & Roadmap Options
# ------------------------------
GRADIENT_ACCUM_STEPS = 8
NUM_FRAMES = 50
PROG_SCHEDULE = [(112, 10), (224, 10), (300, 10)]
LABEL_SMOOTHING = 0.05
USE_FACE_DETECTION = True
PROB_FRAME_DROP = 0.1

# Roadmap flags/options:
LOSS_TYPE = "weighted"         # Options: "weighted", "focal", "ordinal"
USE_OVERSAMPLING = True
TEMPORAL_MODULE = "lstm"       # Options: "lstm", "transformer"
USE_CBAM = True
USE_CROSS_ATTENTION = True
USE_TB_LOGGING = TB_AVAILABLE

# ------------------------------
# Paths & Device
# ------------------------------
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

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
torch.backends.cuda.enable_flash_sdp(True)
torch.multiprocessing.set_start_method('spawn', force=True)

# ------------------------------
# 1) FACE DETECTION HELPER
# ------------------------------
def detect_and_crop_face(img: Image.Image, expand_ratio=0.1):
    cv_img = np.array(img)[:, :, ::-1].copy()  # Convert PIL -> BGR numpy array
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img
    biggest_face = max(faces, key=lambda f: f[2]*f[3])
    x, y, w, h = biggest_face
    x_pad = int(w * expand_ratio)
    y_pad = int(h * expand_ratio)
    x1 = max(0, x - x_pad)
    y1 = max(0, y - y_pad)
    x2 = min(cv_img.shape[1], x + w + x_pad)
    y2 = min(cv_img.shape[0], y + h + y_pad)
    face_crop = cv_img[y1:y2, x1:x2]
    return Image.fromarray(face_crop[:,:,::-1])

# ------------------------------
# 2) DATA AUGMENTATION
# ------------------------------
def get_train_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

def get_val_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

# ------------------------------
# 3) SELECT & PRECOMPUTE FRAMES
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
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    return [frame_files[i] for i in indices]

def precompute_best_frames(csv_file: Path, video_root: Path, num_frames=50, resolution=224):
    data = pd.read_csv(csv_file, dtype=str)
    data.columns = data.columns.str.strip()
    split = csv_file.stem.replace("Labels", "").strip()
    valid_indices = []
    precomputed = []
    skipped = 0
    total_rows = len(data)
    print(f"Precomputing frames for {csv_file.name} at res={resolution} ...")
    for idx, row in tqdm(data.iterrows(), total=total_rows, desc=f"Precompute {csv_file.stem} {resolution}"):
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
    print(f" -> Skipped {skipped} of {len(data)} for {csv_file.name}.")
    cache_data = {"valid_indices": valid_indices, "precomputed_frames": precomputed}
    cache_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    return cache_data

# ------------------------------
# 4) LMDB: CNN FEATURE EXTRACTION
# ------------------------------
def convert_pkl_to_lmdb(csv_file: Path, num_frames=50, resolution=224, lmdb_map_size=1*1024**3, is_train=False):
    pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
    if not pkl_file.exists():
        precompute_best_frames(csv_file, FRAMES_DIR, num_frames, resolution)
    lmdb_path = CACHE_DIR / f"lmdb_new_{csv_file.stem}_frame_{num_frames}_{resolution}"
    if (lmdb_path / "data.mdb").exists():
        print(f"LMDB already exists: {lmdb_path}")
        return lmdb_path
    transform = get_train_transform(resolution) if is_train else get_val_transform(resolution)
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    valid_indices = data["valid_indices"]
    frames_list = data["precomputed_frames"]
    env = lmdb.open(str(lmdb_path), map_size=lmdb_map_size)
    backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
    backbone.reset_classifier(0)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"Converting to LMDB for {csv_file.stem} at res={resolution} (train={is_train})")
    with env.begin(write=True) as txn:
        for idx, paths in enumerate(tqdm(frames_list, desc=f"LMDB {csv_file.stem} {resolution}", leave=True)):
            keep_paths = []
            if is_train:
                for fp in paths:
                    if random.random() < PROB_FRAME_DROP:
                        continue
                    keep_paths.append(fp)
                if len(keep_paths) < len(paths) // 2:
                    keep_paths = paths
            else:
                keep_paths = paths
            all_feats = []
            for fp in keep_paths:
                try:
                    im = Image.open(fp).convert("RGB")
                except:
                    im = Image.new("RGB", (resolution, resolution))
                if USE_FACE_DETECTION and is_train:
                    im = detect_and_crop_face(im, expand_ratio=0.1)
                x_t = transform(im).unsqueeze(0).to(device)
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    feat_map = backbone.forward_features(x_t)
                    feat = F.adaptive_avg_pool2d(feat_map, 1).view(-1)
                    feat = feat.cpu().half()
                all_feats.append(feat.numpy())
            if not all_feats:
                feat_dim = backbone.num_features
                all_feats = [np.zeros((feat_dim,), dtype=np.float16)]
            video_array = np.stack(all_feats, axis=0)
            key = f"video_{valid_indices[idx]}".encode("utf-8")
            txn.put(key, pickle.dumps(video_array))
    env.close()
    print(f"LMDB created at {lmdb_path}")
    return lmdb_path

# ------------------------------
# 5) DATASET WITH OVERSAMPLING OPTION
# ------------------------------
class VideoDatasetLMDB(Dataset):
    def __init__(self, csv_file, lmdb_path, resolution=224, num_frames=50):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}_{resolution}.pkl"
        with open(pkl_file, "rb") as f:
            c = pickle.load(f)
        self.valid_ids = c["valid_indices"]
        self.data = self.data.iloc[self.valid_ids].reset_index(drop=True)
        self.lmdb_path = str(lmdb_path)
        self.env = None
        eng_labels = self.data["Engagement"].astype(int)
        freq = eng_labels.value_counts().to_dict()
        self.sample_weights = [1.0 / freq[label] for label in eng_labels]

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        env = self._init_env()
        key = f"video_{self.valid_ids[idx]}".encode("utf-8")
        with env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise IndexError("Key not found:", key)
            feat_np = pickle.loads(data_bytes)
        feats = torch.from_numpy(feat_np)  # (T, feature_dim)
        # Force each sample to have exactly NUM_FRAMES frames
        T = feats.size(0)
        if T < NUM_FRAMES:
            pad = torch.zeros(NUM_FRAMES - T, feats.size(1), dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=0)
        elif T > NUM_FRAMES:
            feats = feats[:NUM_FRAMES]
        row = self.data.iloc[idx]
        labels = np.array([row["Engagement"], row["Boredom"], row["Confusion"], row["Frustration"]], dtype=np.int64)
        return feats, torch.tensor(labels, dtype=torch.long)

def custom_collate_fn(batch):
    feats, labs = zip(*batch)
    return torch.stack(feats, dim=0), torch.stack(labs, dim=0)

# ------------------------------
# 6) LOSS FUNCTIONS
# ------------------------------
class WeightedLabelSmoothCE(nn.Module):
    def __init__(self, class_weights=None, smoothing=0.05):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        B = inputs.size(0)
        total_loss = 0
        for d in range(4):
            logits = inputs[:, d]  # (B, 4)
            tgt_d = targets[:, d]
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (logits.size(1) - 1))
                for i in range(B):
                    true_dist[i, tgt_d[i]] = 1.0 - self.smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            if self.class_weights is not None:
                cw = self.class_weights.to(inputs.device)
                loss = -(true_dist * (log_probs * cw.unsqueeze(0))).sum(dim=-1)
            else:
                loss = -(true_dist * log_probs).sum(dim=-1)
            total_loss += loss.mean()
        return total_loss / 4.0

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            at = self.alpha[targets].to(inputs.device)
            focal_loss = at * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def ordinal_loss(inputs, targets):
    total_loss = 0
    for d in range(4):
        logits = inputs[:, d]
        prob = F.softmax(logits, dim=-1)
        B, num_classes = prob.shape
        cum_prob = torch.cumsum(prob, dim=-1)
        true_cum = torch.zeros_like(cum_prob)
        for i in range(B):
            k = targets[i, d]
            true_cum[i, :k+1] = 1.0
        loss = F.mse_loss(cum_prob, true_cum, reduction="mean")
        total_loss += loss
    return total_loss / 4.0

def get_criterion(loss_type="weighted"):
    if loss_type == "weighted":
        return WeightedLabelSmoothCE(class_weights=[1,1,1,1], smoothing=LABEL_SMOOTHING).to(device)
    elif loss_type == "focal":
        return FocalLoss(gamma=2, alpha=torch.tensor([1,1,1,1])).to(device)
    elif loss_type == "ordinal":
        return ordinal_loss  # function loss
    else:
        raise ValueError("Invalid loss type")

# ------------------------------
# 7) CBAM & CROSS-ATTENTION MODULES
# ------------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()
        self.conv_s = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = self.sigmoid_c(avg_out + max_out).view(B, C, 1, 1)
        x = x * scale
        avg_c = torch.mean(x, dim=1, keepdim=True)
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_c, max_c], dim=1)
        s_attn = self.sigmoid_s(self.conv_s(cat))
        return x * s_attn

# --- FIX: Use batch_first=True in CrossAttentionModule ---
class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    def forward(self, global_context, frame_feats):
        if not USE_CROSS_ATTENTION:
            return global_context
        # global_context: (B, 1, d_model) and frame_feats: (B, T, d_model)
        out, _ = self.attn(global_context, frame_feats, frame_feats, need_weights=False)
        return out

# ------------------------------
# 8) TEMPORAL MODULES: LSTM & Transformer
# ------------------------------
class TemporalFlashModule(nn.Module):
    def __init__(self, input_dim, lstm_hidden=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
    def forward(self, x):
        # Ensure LSTM gets FP32 input
        x = x.float()
        out, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)
        B, T, E = out.size()
        # Compute global context via mean pooling over T
        global_ctx = out.mean(dim=1)  # (B, E)
        return out, global_ctx

class TemporalTransformerModule(nn.Module):
    def __init__(self, input_dim, d_model=512, n_layers=2, n_heads=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x_proj = self.input_proj(x)  # (B, T, d_model)
        x_t = x_proj.transpose(0,1)   # (T, B, d_model)
        trans_out = self.transformer(x_t)  # (T, B, d_model)
        trans_out = trans_out.transpose(0,1)  # (B, T, d_model)
        global_ctx = self.pool(trans_out.transpose(1,2)).squeeze(-1)  # (B, d_model)
        return trans_out, global_ctx

# ------------------------------
# 9) MODEL WITH ABLATION SWITCHES & FIXED SEQUENCE LENGTH
# ------------------------------
class EfficientNetV2L_Enhanced(nn.Module):
    def __init__(self, lstm_hidden=512, dropout_rate=0.5, n_heads=4):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True)
        self.backbone.reset_classifier(0)
        # Force feature_dim to match our pretrained LMDB extraction (1280)
        self.feature_dim = 1280
        for i, block in enumerate(self.backbone.blocks):
            if i < 6:
                for p in block.parameters():
                    p.requires_grad = False
        if TEMPORAL_MODULE == "lstm":
            self.temporal = TemporalFlashModule(self.feature_dim, lstm_hidden, n_layers=2, dropout=0.3)
        elif TEMPORAL_MODULE == "transformer":
            self.temporal = TemporalTransformerModule(self.feature_dim, d_model=2 * lstm_hidden, n_layers=2, n_heads=n_heads, dropout=0.3)
        else:
            raise ValueError("Invalid TEMPORAL_MODULE")
        self.cbam = CBAMBlock(2 * lstm_hidden, reduction=16) if USE_CBAM else nn.Identity()
        self.cross_attn = CrossAttentionModule(d_model=2 * lstm_hidden, n_heads=n_heads)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_eng = nn.Linear(2 * lstm_hidden, 4)
        self.fc_bor = nn.Linear(2 * lstm_hidden, 4)
        self.fc_con = nn.Linear(2 * lstm_hidden, 4)
        self.fc_fru = nn.Linear(2 * lstm_hidden, 4)
    def forward(self, x):
        # x: (B, T, feature_dim) -- collate_fn ensures T == NUM_FRAMES
        out, global_ctx = self.temporal(x)  # out: (B, T, E), global_ctx: (B, E)
        B, T, E = out.size()
        # Force T to be exactly NUM_FRAMES
        if T != NUM_FRAMES:
            if T > NUM_FRAMES:
                out = out[:, :NUM_FRAMES, :]
            else:
                pad_size = NUM_FRAMES - T
                padding = torch.zeros(B, pad_size, E, device=out.device, dtype=out.dtype)
                out = torch.cat([out, padding], dim=1)
            T = NUM_FRAMES
        out = out.contiguous()
        # Apply CBAM: reshape to (B, E, T, 1)
        out_reshaped = out.transpose(1,2).unsqueeze(-1).contiguous()  # (B, E, T, 1)
        out_cbam = self.cbam(out_reshaped)  # (B, E, T, 1)
        out_cbam = out_cbam.squeeze(-1).transpose(1,2).contiguous()  # (B, T, E)
        # Prepare query from global context: (B, 1, E)
        query = global_ctx.unsqueeze(1) if global_ctx.dim() == 2 else global_ctx
        query = query.contiguous()
        # Apply cross-attention (with batch_first=True)
        cross_ctx = self.cross_attn(query, out_cbam).squeeze(1)  # (B, E)
        x_drop = self.dropout(cross_ctx)
        eng = self.fc_eng(x_drop)
        bor = self.fc_bor(x_drop)
        con = self.fc_con(x_drop)
        fru = self.fc_fru(x_drop)
        return eng, bor, con, fru

# ------------------------------
# 10) TRAINING FUNCTION WITH TENSORBOARD LOGGING
# ------------------------------
def progressive_train_model(model, total_epochs, initial_lr, checkpoint_path,
                            batch_size, patience=5, unfreeze_after=5,
                            use_swa=True, max_lr_factor=10.0, criterion=None):
    if criterion is None:
        criterion = get_criterion(LOSS_TYPE)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": initial_lr * 0.1},
        {"params": other_params, "lr": initial_lr}
    ], weight_decay=1e-4)
    scaler = GradScaler()
    best_val_loss = float('inf')
    early_count = 0
    current_epoch = 0
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_started = False
    model.to(device)
    writer = SummaryWriter(log_dir=str(BASE_DIR / "logs")) if USE_TB_LOGGING else None
    for res, ep_this_stage in PROG_SCHEDULE:
        train_lmdb = convert_pkl_to_lmdb(train_csv, NUM_FRAMES, res, is_train=True)
        val_lmdb = convert_pkl_to_lmdb(val_csv, NUM_FRAMES, res, is_train=False)
        train_ds = VideoDatasetLMDB(train_csv, train_lmdb, resolution=res)
        val_ds = VideoDatasetLMDB(val_csv, val_lmdb, resolution=res)
        if USE_OVERSAMPLING:
            sampler = WeightedRandomSampler(train_ds.sample_weights, num_samples=len(train_ds), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        steps_per_epoch = len(train_loader)
        max_lr = initial_lr * max_lr_factor
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, [initial_lr, max_lr],
            steps_per_epoch=steps_per_epoch,
            epochs=ep_this_stage,
            pct_start=0.3
        )
        if use_swa:
            swa_scheduler = SWALR(optimizer, swa_lr=initial_lr * 0.1)
        for e in range(ep_this_stage):
            model.train()
            running_loss = 0.0
            ep_desc = f"Epoch {current_epoch+1}/{total_epochs} (res={res})"
            pbar = tqdm(train_loader, desc=ep_desc, leave=False)
            for i, (feats, labs) in enumerate(pbar):
                feats = feats.to(device)
                labs = labs.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    eng, bor, con, fru = model(feats)
                    outs = torch.stack([eng, bor, con, fru], dim=1)
                    loss = criterion(outs, labs)
                scaler.scale(loss).backward()
                if ((i + 1) % GRADIENT_ACCUM_STEPS == 0) or (i + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item() * feats.size(0)
                scheduler.step()
                pbar.set_postfix({"loss": loss.item()})
            train_loss = running_loss / len(train_loader.dataset)
            if current_epoch >= unfreeze_after:
                for i, block in enumerate(model.backbone.blocks):
                    if i < 6:
                        for p in block.parameters():
                            p.requires_grad = True
            if use_swa and (current_epoch >= total_epochs // 2):
                swa_model.update_parameters(model)
                swa_started = True
            model.eval()
            val_sum = 0.0
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                for feats, labs in val_loader:
                    feats = feats.to(device)
                    labs = labs.to(device)
                    eng, bor, con, fru = model(feats)
                    outs = torch.stack([eng, bor, con, fru], dim=1)
                    vloss = criterion(outs, labs)
                    val_sum += vloss.item() * feats.size(0)
            val_loss = val_sum / len(val_loader.dataset)
            print(f"[res={res}] Epoch {current_epoch+1}/{total_epochs} | Train={train_loss:.4f} | Val={val_loss:.4f}")
            if writer:
                writer.add_scalar("Loss/Train", train_loss, current_epoch)
                writer.add_scalar("Loss/Val", val_loss, current_epoch)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], current_epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {"epoch": current_epoch + 1,
                        "val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}
                torch.save(ckpt, checkpoint_path)
                early_count = 0
            else:
                early_count += 1
            current_epoch += 1
            if early_count >= patience:
                print(f"Early stop, best val={best_val_loss:.4f}")
                if use_swa and swa_started:
                    from torch.optim.swa_utils import update_bn
                    update_bn(train_loader, swa_model, device=device)
                    ckpt = {"epoch": current_epoch,
                            "val_loss": best_val_loss,
                            "model_state_dict": swa_model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}
                    torch.save(ckpt, checkpoint_path)
                if writer:
                    writer.close()
                return best_val_loss
        if use_swa and swa_started:
            swa_scheduler.step()
            from torch.optim.swa_utils import update_bn
            update_bn(train_loader, swa_model, device=device)
            model.load_state_dict(swa_model.module.state_dict())
    if writer:
        writer.close()
    return best_val_loss

# ------------------------------
# 11) EVALUATION
# ------------------------------
def evaluate_model(model, test_loader, num_tta=3):
    model.eval()
    all_preds = {0: [], 1: [], 2: [], 3: []}
    all_labels = {0: [], 1: [], 2: [], 3: []}
    pbar = tqdm(test_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for feats, labs in pbar:
            feats = feats.to(device)
            labs = labs.to(device)
            eng_acc = 0; bor_acc = 0; con_acc = 0; fru_acc = 0
            for _ in range(num_tta):
                eng, bor, con, fru = model(feats)
                eng_acc += torch.softmax(eng, dim=1)
                bor_acc += torch.softmax(bor, dim=1)
                con_acc += torch.softmax(con, dim=1)
                fru_acc += torch.softmax(fru, dim=1)
            out_eng = eng_acc / num_tta
            out_bor = bor_acc / num_tta
            out_con = con_acc / num_tta
            out_fru = fru_acc / num_tta
            all_preds[0].append(torch.argmax(out_eng, dim=1).cpu())
            all_preds[1].append(torch.argmax(out_bor, dim=1).cpu())
            all_preds[2].append(torch.argmax(out_con, dim=1).cpu())
            all_preds[3].append(torch.argmax(out_fru, dim=1).cpu())
            all_labels[0].append(labs[:, 0].cpu())
            all_labels[1].append(labs[:, 1].cpu())
            all_labels[2].append(labs[:, 2].cpu())
            all_labels[3].append(labs[:, 3].cpu())
    for k in all_preds:
        all_preds[k] = torch.cat(all_preds[k], dim=0).numpy()
        all_labels[k] = torch.cat(all_labels[k], dim=0).numpy()
    states = ["Engagement", "Boredom", "Confusion", "Frustration"]
    for i, st in enumerate(states):
        print(f"== {st} ==")
        print(classification_report(all_labels[i], all_preds[i],
                                    target_names=["Very Low", "Low", "High", "Very High"], digits=3))
        cm = confusion_matrix(all_labels[i], all_preds[i], labels=[0, 1, 2, 3])
        plt.figure(figsize=(6,5))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {st}")
        plt.colorbar()
        plt.xticks(np.arange(4), ["VL", "L", "H", "VH"])
        plt.yticks(np.arange(4), ["VL", "L", "H", "VH"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv   = LABELS_DIR / "ValidationLabels.csv"
    test_csv  = LABELS_DIR / "TestLabels.csv"

    # Precompute frames and convert to LMDB (reuse if they exist)
    for csvp in [train_csv, val_csv, test_csv]:
        for rs in [112, 224, 300]:
            precompute_best_frames(csvp, FRAMES_DIR, NUM_FRAMES, rs)
    for csvp in [train_csv, val_csv, test_csv]:
        for rs in [112, 224, 300]:
            convert_pkl_to_lmdb(csvp, NUM_FRAMES, rs, is_train=(csvp == train_csv))

    # Hyperparameter Tuning with Optuna
    db_path = BASE_DIR / "notebooks" / "effv2l_bilstm_3res.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        print(f"Database created/connected at {db_path}")
        conn.close()
    except Exception as e:
        print(e)

    study = optuna.create_study(direction="minimize",
                                study_name="effv2l_3stage_tqdmstudy",
                                storage=f"sqlite:///{db_path}",
                                load_if_exists=True,
                                pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=10))

    def objective(trial):
        bs = trial.suggest_categorical("batch_size", [4, 8])
        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        hid = trial.suggest_categorical("lstm_hidden", [512, 768])
        dr = trial.suggest_float("dropout_rate", 0.3, 0.6)
        heads = trial.suggest_categorical("n_heads", [4, 8])
        total_epochs = sum([e for _, e in PROG_SCHEDULE])
        model = EfficientNetV2L_Enhanced(lstm_hidden=hid, dropout_rate=dr, n_heads=heads).to(device)
        ckpt = MODEL_DIR / f"trial_effv2l_bilstm_cbam_3res_{trial.number}_checkpoint.pth"
        val_loss = progressive_train_model(model, total_epochs, lr, ckpt,
                                           batch_size=bs,
                                           patience=3,
                                           unfreeze_after=5,
                                           use_swa=False,
                                           max_lr_factor=5.0,
                                           criterion=get_criterion(LOSS_TYPE))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return val_loss

    target_trials = 30
    successes = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
    remaining = target_trials - len(successes)
    for _ in tqdm(range(remaining), desc="Optuna Trials"):
        study.optimize(objective, n_trials=1, catch=(Exception,))
    successes = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
    if len(successes) == 0:
        print("No successful trials found. Using fallback hyperparams.")
        class FakeTrial:
            params = {"batch_size": 4, "lr": 1e-4, "lstm_hidden": 512, "dropout_rate": 0.5, "n_heads": 4}
        best_trial = FakeTrial()
    else:
        best_trial = min(successes, key=lambda t: t.value)
    print(f"Best trial: {best_trial.params}")
    best_bs = best_trial.params.get("batch_size", 4)
    best_lr = best_trial.params.get("lr", 1e-4)
    best_hid = best_trial.params.get("lstm_hidden", 512)
    best_dr = best_trial.params.get("dropout_rate", 0.5)
    best_hd = best_trial.params.get("n_heads", 4)

    # Final training
    final_model = EfficientNetV2L_Enhanced(lstm_hidden=best_hid, dropout_rate=best_dr, n_heads=best_hd).to(device)
    total_epochs = sum([e for _, e in PROG_SCHEDULE])
    final_ckpt = MODEL_DIR / "final_effv2l_bilstm_cbam_3res_model.pth"
    best_val_loss = progressive_train_model(final_model, total_epochs, best_lr, final_ckpt,
                                            batch_size=best_bs,
                                            patience=5,
                                            unfreeze_after=5,
                                            use_swa=True,
                                            max_lr_factor=10.0,
                                            criterion=get_criterion(LOSS_TYPE))
    print("Final training complete. Best val loss=%.4f" % best_val_loss)

    # Evaluate at resolution 300
    test_lmdb = CACHE_DIR / f"lmdb_new_{test_csv.stem}_frame_{NUM_FRAMES}_300"
    test_ds = VideoDatasetLMDB(test_csv, test_lmdb, resolution=300, num_frames=NUM_FRAMES)
    test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
    st = torch.load(final_ckpt, map_location=device)
    final_model.load_state_dict(st["model_state_dict"])
    final_model.eval()
    evaluate_model(final_model, test_loader, num_tta=3)
