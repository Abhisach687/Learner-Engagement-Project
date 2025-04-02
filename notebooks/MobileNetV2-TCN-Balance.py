import os
import cv2
import gc
import json
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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.storages import RDBStorage
import logging
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)
import sqlite3
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
import lmdb
import copy
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

# Define paths
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

import os
print("Models directory exists:", os.path.exists("C:/Users/abhis/Downloads/Documents/Learner Engagement Project/models"))
print("Checkpoint path writable:", os.access("C:/Users/abhis/Downloads/Documents/Learner Engagement Project/models", os.W_OK))
os.makedirs("C:/Users/abhis/Downloads/Documents/Learner Engagement Project/models", exist_ok=True)

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # for better error traces 

# Precomputed directory for caching best frames
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

torch.backends.cudnn.benchmark = True  # Optimize cuDNN
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32

# Set device and CUDA configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define data transforms for training and validation.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping CSV clip IDs.
def get_csv_clip_id(video_stem: str) -> str:
    base = video_stem.strip()
    if base.startswith("110001"):
        base = base.replace("110001", "202614", 1)
    return base

# Select best frames using face detection and Laplacian variance.
def select_impactful_frames(video_folder: Path, num_frames=30):
    frame_files = sorted(video_folder.glob("frame_*.jpg"))
    total_frames = len(frame_files)
    if total_frames == 0:
        return []
    if total_frames <= num_frames:
        return frame_files
    segment_size = total_frames // num_frames
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    selected_frames = []
    for i in range(num_frames):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_frames - 1 else total_frames
        best_score = -1
        best_frame = None
        for fp in frame_files[start_idx:end_idx]:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                face = max(faces, key=lambda r: r[2]*r[3])
                x, y, w, h = face
                region = gray[y:y+h, x:x+w]
                quality = cv2.Laplacian(region, cv2.CV_64F).var()
            else:
                quality = cv2.Laplacian(gray, cv2.CV_64F).var()
            if quality > best_score:
                best_score = quality
                best_frame = fp
        if best_frame is not None:
            selected_frames.append(best_frame)
    return selected_frames

def precompute_best_frames(csv_file: Path, video_root: Path, num_frames=30):
    """
    Precompute and cache the best frame paths for each video in the CSV.
    The results are saved to a pickle file and returned.
    """
    data = pd.read_csv(csv_file, dtype=str)
    data.columns = data.columns.str.strip()
    split = Path(csv_file).stem.replace("Labels", "").strip()
    precomputed = []
    valid_indices = []
    skipped_count = 0

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Precomputing best frames", dynamic_ncols=True):
        clip_id = str(row["ClipID"]).strip()
        if clip_id.endswith(('.avi', '.mp4')):
            clip_id = clip_id.rsplit('.', 1)[0]
        mapped_id = get_csv_clip_id(clip_id)
        video_folder = video_root / split / mapped_id
        if video_folder.exists():
            frame_files = sorted(video_folder.glob("frame_*.jpg"))
            if len(frame_files) >= num_frames:
                selected_frames = select_impactful_frames(video_folder, num_frames)
                precomputed.append(selected_frames)
                valid_indices.append(idx)
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    print(f"Precomputation: Skipped {skipped_count} videos out of {len(data)}.")
    cache_data = {"valid_indices": valid_indices, "precomputed_frames": precomputed}
    cache_file = CACHE_DIR / f"precomputed_{Path(csv_file).stem}_frame_{num_frames}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Precomputed results saved to {cache_file}")
    return cache_data

# Define the custom Dataset for video classification
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_root, transform=None, num_frames=30):
        self.cache = {}
        self.max_cache_size = 400
        self.cache_order = []
        self.csv_file = Path(csv_file)
        self.data = pd.read_csv(self.csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        self.video_root = Path(video_root)
        self.transform = transform
        self.num_frames = num_frames
        self.split = self.csv_file.stem.replace("Labels", "").strip()
        cache_file = CACHE_DIR / f"precomputed_{Path(csv_file).stem}_frame_{num_frames}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            valid_indices = cache_data["valid_indices"]
            self.precomputed_frames = cache_data["precomputed_frames"]
            self.data = self.data.iloc[valid_indices].reset_index(drop=True)
            print(f"Loaded precomputed frames for {len(self.data)} videos from cache.")
        else:
            valid_rows = []
            self.precomputed_frames = []
            skipped_count = 0
            for idx, row in self.data.iterrows():
                clip_id = str(row["ClipID"]).strip()
                if clip_id.endswith(('.avi', '.mp4')):
                    clip_id = clip_id.rsplit('.', 1)[0]
                mapped_id = get_csv_clip_id(clip_id)
                video_folder = self.video_root / self.split / mapped_id
                if video_folder.exists():
                    frame_files = sorted(video_folder.glob("frame_*.jpg"))
                    if len(frame_files) >= num_frames:
                        selected_frames = select_impactful_frames(video_folder, num_frames)
                        valid_rows.append(row)
                        self.precomputed_frames.append(selected_frames)
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            self.data = pd.DataFrame(valid_rows)
            print(f"Computed frames on the fly: Skipped {skipped_count} videos.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            cached_frames, labels = self.cache[idx]
            return cached_frames.clone(), labels.clone()
        
        row = self.data.iloc[idx]
        selected_frame_paths = self.precomputed_frames[idx]
        
        raw_images = []
        for fp in selected_frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except (FileNotFoundError, OSError):
                img = Image.new('RGB', (224, 224))
            raw_images.append(img)
        
        while len(raw_images) < self.num_frames:
            raw_images.append(Image.new('RGB', (224, 224)))
        
        transformed_frames = [self.transform(img) for img in raw_images]
        
        frames_tensor = torch.stack(transformed_frames)
        labels = torch.tensor([
            int(row["Engagement"]),
            int(row["Boredom"]),
            int(row["Confusion"]),
            int(row["Frustration"])
        ], dtype=torch.long)
        
        # Cache only if not using random transforms (validation/test)
        if not isinstance(self.transform, transforms.Compose) or \
           not any(isinstance(t, transforms.RandomHorizontalFlip) for t in self.transform.transforms):
            self.cache[idx] = (frames_tensor.cpu(), labels.cpu())
        
        if not isinstance(self.transform, transforms.Compose) or \
           not any(isinstance(t, transforms.RandomHorizontalFlip) for t in self.transform.transforms):
            if len(self.cache) >= self.max_cache_size:
                oldest_idx = self.cache_order.pop(0)
                del self.cache[oldest_idx]
            self.cache[idx] = (frames_tensor.cpu(), labels.cpu())
            self.cache_order.append(idx)
        
        return frames_tensor, labels

# Define the MobileNetV2-TCN model
class EmotionSpecialistMobileTCN(nn.Module):
    def __init__(self, hidden_dim=192, dropout_rate=0.25):
        super().__init__()
        # MobileNetV2 backbone
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet.classifier = nn.Identity()
        
        # Emotion-specific feature extractors (one per emotion)
        self.emotion_extractors = nn.ModuleDict({
            'engagement': nn.Sequential(
                nn.Conv1d(1280, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ),
            'boredom': nn.Sequential(
                nn.Conv1d(1280, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ),
            'confusion': nn.Sequential(
                nn.Conv1d(1280, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ),
            'frustration': nn.Sequential(
                nn.Conv1d(1280, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        })
        
        # Temporal attention for each emotion
        self.attention = nn.ModuleDict({
            'engagement': nn.MultiheadAttention(hidden_dim, 4, batch_first=True),
            'boredom': nn.MultiheadAttention(hidden_dim, 4, batch_first=True),
            'confusion': nn.MultiheadAttention(hidden_dim, 4, batch_first=True),
            'frustration': nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        })
        
        # Final classifiers
        self.classifiers = nn.ModuleDict({
            'engagement': self._make_classifier(hidden_dim),
            'boredom': self._make_classifier(hidden_dim),
            'confusion': self._make_classifier(hidden_dim),
            'frustration': self._make_classifier(hidden_dim)
        })
        
    def _make_classifier(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.mobilenet(x)
        features = features.view(batch_size, seq_len, -1)
        features = features.permute(0, 2, 1)  # [B, 1280, seq_len]
        
        outputs = []
        emotion_names = ['engagement', 'boredom', 'confusion', 'frustration']
        
        for emotion in emotion_names:
            # Extract emotion-specific features
            emotion_features = self.emotion_extractors[emotion](features)  # [B, hidden_dim, seq_len]
            emotion_features = emotion_features.permute(0, 2, 1)  # [B, seq_len, hidden_dim]
            
            # Apply attention
            attn_features, _ = self.attention[emotion](
                emotion_features, emotion_features, emotion_features
            )
            
            # Global pooling
            pooled_features = torch.mean(attn_features, dim=1)  # [B, hidden_dim]
            
            # Classification
            emotion_output = self.classifiers[emotion](pooled_features)  # [B, 4]
            outputs.append(emotion_output)
            
        return torch.stack(outputs, dim=1)  # [B, 4, 4]

# Define training, checkpointing, and evaluation functions
def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path):
    try:
        print(f"Saving checkpoint to {checkpoint_path} ...")  # Debugging print
        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temp_path)
        
        # Remove existing file if it exists
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
        # Rename temp file
        temp_path.rename(checkpoint_path)
        
        print(f"Checkpoint saved successfully to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise

def load_checkpoint(model, optimizer, checkpoint_path):
    if checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            return state["epoch"], state["best_val_loss"]
        except:
            print(f"Error loading checkpoint {checkpoint_path}, starting from scratch")
            return 0, float("inf")
    return 0, float("inf")


class ExtremeClassBalancingLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=0.25):
        super().__init__()
        # These enormous weights counter extreme class imbalance
        self.class_weights = {
            'engagement': torch.tensor([25.0, 20.0, 1.0, 5.0]),
            'boredom': torch.tensor([2.0, 3.0, 12.0, 20.0]),
            'confusion': torch.tensor([1.5, 3.0, 15.0, 25.0]),
            'frustration': torch.tensor([1.0, 0.2, 20.0, 25.0])
        }
        self.gamma = gamma
        self.alpha = alpha
        
        # Target class distributions (aggressive forcing of minority classes)
        self.target_distributions = {
            'engagement': torch.tensor([0.25, 0.25, 0.25, 0.25]),
            'boredom': torch.tensor([0.25, 0.25, 0.25, 0.25]),
            'confusion': torch.tensor([0.25, 0.25, 0.25, 0.25]),
            'frustration': torch.tensor([0.25, 0.25, 0.25, 0.25])
        }
        self.distribution_weight = 2.0
    
    def forward(self, outputs, targets, return_components=False):
        batch_size = outputs.size(0)
        emotion_names = ['engagement', 'boredom', 'confusion', 'frustration']
        
        focal_loss = 0
        dist_loss = 0
        
        for i, emotion in enumerate(emotion_names):
            logits = outputs[:, i]
            target = targets[:, i]
            weights = self.class_weights[emotion].to(logits.device)
            
            # One-hot encode targets
            target_one_hot = F.one_hot(target, num_classes=4).float()
            
            # Apply label smoothing (0.1)
            smoothed_targets = target_one_hot * 0.9 + 0.025
            
            # Calculate focal loss component
            probs = F.softmax(logits, dim=1)
            pt = torch.sum(smoothed_targets * probs, dim=1)
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply class weights
            class_weights = torch.matmul(target_one_hot, weights.unsqueeze(1)).squeeze()
            
            # Calculate loss
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -torch.sum(smoothed_targets * log_probs, dim=1)
            emotion_loss = focal_weight * class_weights * ce_loss
            focal_loss += emotion_loss.mean()
            
            # Distribution matching loss
            pred_dist = F.softmax(logits, dim=1).mean(0)
            target_dist = self.target_distributions[emotion].to(pred_dist.device)
            dist_loss += F.kl_div(pred_dist.log(), target_dist, reduction='sum')
        
        # Combine losses
        total_loss = focal_loss + self.distribution_weight * dist_loss
        
        if return_components:
            return total_loss, focal_loss, dist_loss
        return total_loss
    
    
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average logits from all models
        return torch.mean(torch.stack(outputs), dim=0)
    
    
def plot_training_progress(history):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    
    # Plot accuracy curves
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    
    # Plot F1 scores
    axes[1, 0].plot(history['val_f1_macro'], label='Macro F1')
    for emotion in ['engagement', 'boredom', 'confusion', 'frustration']:
        axes[1, 0].plot(history[f'val_f1_{emotion}'], label=f'{emotion.capitalize()} F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Scores')
    axes[1, 0].legend()
    
    # Plot class collapse indicator
    collapse_scores = []
    for epoch in range(len(history['val_loss'])):
        collapse_score = 0
        for emotion in ['engagement', 'boredom', 'confusion', 'frustration']:
            dist = history[f'val_dist_{emotion}'][epoch]
            min_class_pct = min(dist)
            collapse_score += (0.25 - min_class_pct) / 0.25  # Score based on deviation from ideal
        collapse_scores.append(collapse_score / 4)  # Average across emotions
    
    axes[1, 1].plot(collapse_scores, 'r-', label='Class Collapse Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Class Collapse Score (0-1)')
    axes[1, 1].set_title('Class Collapse Indicator (lower is better)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300)
    plt.show()



def train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch, accumulation_steps=2):
    model.train()
    running_loss = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", dynamic_ncols=True)
    
    for i, (frames, labels) in enumerate(pbar):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
            outputs = model(frames)
            loss = loss_fn(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        # Only update weights after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * frames.size(0) * accumulation_steps
        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
        
        if (i + 1) % 15 == 0:  # Clear cache more often
            torch.cuda.empty_cache()
    
    # Handle remaining gradients at the end of epoch
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, loss_fn, epoch):
    model.eval()
    running_loss = 0
    emotion_names = ['engagement', 'boredom', 'confusion', 'frustration']
    all_targets = {e: [] for e in emotion_names}
    all_preds = {e: [] for e in emotion_names}
    all_probs = {e: [] for e in emotion_names}
    
    with torch.no_grad(), autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
        for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", dynamic_ncols=True):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(frames)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * frames.size(0)
            
            # Track predictions and targets
            for i, emotion in enumerate(emotion_names):
                probs = F.softmax(outputs[:, i], dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds[emotion].append(preds.cpu())
                all_targets[emotion].append(labels[:, i].cpu())
                all_probs[emotion].append(probs.cpu())
    
    # Calculate metrics
    metrics = {}
    val_loss = running_loss / len(val_loader.dataset)
    metrics['val_loss'] = val_loss
    
    avg_f1 = 0
    for i, emotion in enumerate(emotion_names):
        preds = torch.cat(all_preds[emotion]).numpy()
        targets = torch.cat(all_targets[emotion]).numpy()
        probs = torch.cat(all_probs[emotion], dim=0)
        
        # Calculate prediction distribution
        pred_dist = torch.bincount(torch.tensor(preds), minlength=4).float() / len(preds)
        metrics[f'val_dist_{emotion}'] = pred_dist.tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        f1_macro = f1_score(targets, preds, average='macro')
        f1_per_class = f1_score(targets, preds, average=None)
        
        metrics[f'accuracy_{emotion}'] = accuracy
        metrics[f'f1_macro_{emotion}'] = f1_macro
        metrics[f'f1_per_class_{emotion}'] = f1_per_class.tolist()
        
        avg_f1 += f1_macro
        
        # Check for class collapse
        min_class_pct = torch.min(pred_dist).item()
        if min_class_pct < 0.05:
            print(f"WARNING: {emotion} shows class collapse: {pred_dist.tolist()}")
        
        # Print metrics
        print(f"\n{emotion.capitalize()} metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  F1 per class: {f1_per_class}")
        print(f"  Prediction distribution: {pred_dist.tolist()}")
    
    avg_f1 /= len(emotion_names)
    metrics['avg_accuracy'] = sum(metrics[f'accuracy_{e}'] for e in emotion_names) / len(emotion_names)
    metrics['avg_f1'] = avg_f1
    
    print(f"\nOverall metrics:")
    print(f"  Average Accuracy: {metrics['avg_accuracy']:.4f}")
    print(f"  Average F1 Macro: {metrics['avg_f1']:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    return val_loss, metrics, avg_f1

def train_model_with_phases(model, train_loader, val_loader, checkpoint_path, num_epochs=30):
    model.to(device, non_blocking=True)
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1_macro': [],
        'val_f1_engagement': [], 'val_f1_boredom': [], 'val_f1_confusion': [], 'val_f1_frustration': [],
        'val_dist_engagement': [], 'val_dist_boredom': [], 'val_dist_confusion': [], 'val_dist_frustration': []
    }
    
    best_f1 = 0
    best_model_state = None
    patience = 7
    patience_counter = 0
    
    # Phase 1: Feature extraction (5 epochs)
    print("\n===== Phase 1: Feature learning =====")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = GradScaler()
    
    for param in model.mobilenet.parameters():
        param.requires_grad = True
    
    loss_fn = ExtremeClassBalancingLoss(gamma=2.0, alpha=0.25)
    loss_fn.distribution_weight = 0.5  # Lower distribution weight in phase 1
    
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch)
        val_loss, metrics, avg_f1 = validate(model, val_loader, loss_fn, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(metrics['avg_accuracy'])
        history['val_f1_macro'].append(metrics['avg_f1'])
        for e in ['engagement', 'boredom', 'confusion', 'frustration']:
            history[f'val_f1_{e}'].append(metrics[f'f1_macro_{e}'])
            history[f'val_dist_{e}'].append(metrics[f'val_dist_{e}'])
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
    
    # Phase 2: Specialist training (10 epochs)
    print("\n===== Phase 2: Specialist training =====")
    # Freeze backbone
    for param in model.mobilenet.parameters():
        param.requires_grad = False
    
    # Increase specialized weights even more
    loss_fn = ExtremeClassBalancingLoss(gamma=3.0, alpha=0.5)
    loss_fn.distribution_weight = 2.0  # Higher distribution weight
    
    for emotion in ['engagement', 'boredom', 'confusion', 'frustration']:
        loss_fn.class_weights[emotion] *= 1.5  # Increase weights
    
    optimizer = optim.AdamW([
        {'params': model.emotion_extractors.parameters(), 'lr': 2e-4},
        {'params': model.attention.parameters(), 'lr': 2e-4},
        {'params': model.classifiers.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    patience_counter = 0
    
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch + 5)
        val_loss, metrics, avg_f1 = validate(model, val_loader, loss_fn, epoch + 5)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(metrics['avg_accuracy'])
        history['val_f1_macro'].append(metrics['avg_f1'])
        for e in ['engagement', 'boredom', 'confusion', 'frustration']:
            history[f'val_f1_{e}'].append(metrics[f'f1_macro_{e}'])
            history[f'val_dist_{e}'].append(metrics[f'val_dist_{e}'])
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch + 1 + 5, val_loss, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 5}")
            break
            
        scheduler.step()
    
    # Phase 3: Fine-tuning (up to 15 epochs)
    print("\n===== Phase 3: Fine-tuning =====")
    # Load best model state so far
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Unfreeze backbone with small learning rate
    for param in model.mobilenet.parameters():
        param.requires_grad = True
    
    # Maximize distribution forcing
    loss_fn = ExtremeClassBalancingLoss(gamma=3.0, alpha=0.5)
    loss_fn.distribution_weight = 3.0
    
    optimizer = optim.AdamW([
        {'params': model.mobilenet.parameters(), 'lr': 5e-6},
        {'params': model.emotion_extractors.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.classifiers.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    patience_counter = 0
    
    for epoch in range(15):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch + 15)
        val_loss, metrics, avg_f1 = validate(model, val_loader, loss_fn, epoch + 15)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(metrics['avg_accuracy'])
        history['val_f1_macro'].append(metrics['avg_f1'])
        for e in ['engagement', 'boredom', 'confusion', 'frustration']:
            history[f'val_f1_{e}'].append(metrics[f'f1_macro_{e}'])
            history[f'val_dist_{e}'].append(metrics[f'val_dist_{e}'])
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch + 1 + 15, val_loss, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 15}")
            break
            
        scheduler.step()
    
    # Plot training progress
    plot_training_progress(history)
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, best_f1, history

# Define the Optuna objective for hyperparameter tuning (using SQLite storage)
def objective(trial):
    num_frames = trial.suggest_categorical("num_frames", [30])
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [96, 128, 192])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)

    train_dataset = VideoDataset(LABELS_DIR / "TrainLabels.csv", FRAMES_DIR, transform=train_transform, num_frames=num_frames)
    val_dataset = VideoDataset(LABELS_DIR / "ValidationLabels.csv", FRAMES_DIR, transform=val_transform, num_frames=num_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    
    trial_checkpoint = MODEL_DIR / f"checkpoint_trial_balance_{trial.number}.pth"
    trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    try:
        loss_fn = ExtremeClassBalancingLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler()
        model.to(device)
        
        # Just run phase 1 training for trials
        train_loss = 0
        val_loss = 0
        
        for epoch in range(3):  # Reduced epochs for trial
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch)
            val_loss, metrics, avg_f1 = validate(model, val_loader, loss_fn, epoch)
            
        return -metrics['avg_f1']  # Negative for minimization
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")
    
# Evaluate and visualize results
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_loader):
    # Check if this is an EnsembleModel to use memory-efficient evaluation
    is_ensemble = isinstance(model, EnsembleModel)
    
    if is_ensemble:
        # print("Using memory-efficient ensemble evaluation...")
        # models = model.models
        # num_models = len(models)
        
        # # Setup for accumulating predictions
        # all_labels = None
        # accumulated_logits = None
        
        # # Process each model separately to save memory
        # for model_idx, single_model in enumerate(models):
        #     print(f"Evaluating model {model_idx+1}/{num_models}")
        #     single_model.eval()
            
        #     batch_logits = []
        #     batch_labels = []
            
        #     with torch.no_grad(), autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
        #         for frames, labels in tqdm(test_loader, desc=f"Model {model_idx+1}", dynamic_ncols=True):
        #             frames = frames.to(device, non_blocking=True)
        #             outputs = single_model(frames)
                    
        #             batch_logits.append(outputs.cpu())
        #             if model_idx == 0:  # Only need to save labels once
        #                 batch_labels.append(labels)
            
        #     # Process this model's results
        #     model_logits = torch.cat(batch_logits, dim=0)
            
        #     if accumulated_logits is None:
        #         accumulated_logits = model_logits
        #         if model_idx == 0:
        #             all_labels = torch.cat(batch_labels, dim=0).numpy()
        #     else:
        #         accumulated_logits += model_logits
            
        #     # Clear memory after processing each model
        #     torch.cuda.empty_cache()
            
        #     # If this isn't the last model, remove it from GPU to save memory
        #     if model_idx < num_models - 1:
        #         single_model.cpu()
        #         torch.cuda.empty_cache()
        
        # # Average the accumulated logits
        # all_logits = accumulated_logits / num_models
        # all_preds = torch.argmax(all_logits, dim=2).numpy()
        pass
        
    else:
        # Standard single model evaluation
        model.eval()
        batch_all_preds = []
        batch_all_labels = []
        total_batches = len(test_loader)
        
        with torch.no_grad(), autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
            for frames, labels in tqdm(test_loader, desc="Evaluating", total=total_batches, dynamic_ncols=True):
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(frames)
                
                batch_all_labels.append(labels.cpu())
                batch_all_preds.append(outputs.cpu())
                
                # Free up memory
                if (len(batch_all_preds) % 5 == 0):
                    torch.cuda.empty_cache()
            
            # Process all batches together
            all_labels = torch.cat(batch_all_labels, dim=0).numpy()
            all_logits = torch.cat(batch_all_preds, dim=0)
            all_preds = torch.argmax(all_logits, dim=2).numpy()
    
    # Continue with evaluation code - metrics calculation and visualization
    results = {}
    metrics = ["Engagement", "Boredom", "Confusion", "Frustration"]
    
    for i, metric in enumerate(metrics):
        print(f"\nEvaluating model for {metric}...")
        
        # Calculate accuracy
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        f1 = f1_score(all_labels[:, i], all_preds[:, i], average='macro')
        results[metric] = {'accuracy': acc, 'f1': f1}
        print(f"{metric} validation accuracy: {acc:.4f}")
        print(f"{metric} validation F1-score: {f1:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(all_labels[:, i], all_preds[:, i])
        print(report)
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        print("Confusion Matrix:")
        print(cm)
        
        # --- Visualizations ---
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{metric} - Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels") 
        plt.tight_layout()
        plt.savefig(f"{metric}_confusion_matrix.png", dpi=300)
        plt.show()
        
        # 2. Bar Chart for Label Distribution
        true_counts = np.bincount(all_labels[:, i], minlength=4)
        pred_counts = np.bincount(all_preds[:, i], minlength=4)
        
        plt.figure(figsize=(8, 4))
        width = 0.35
        x = np.arange(4)
        plt.bar(x - width/2, true_counts, width, label="True Labels")
        plt.bar(x + width/2, pred_counts, width, label="Predicted Labels")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(f"{metric} - Distribution of True vs Predicted Labels")
        plt.xticks(x, x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_label_distribution.png", dpi=300)
        plt.show()
        
        # 3. F1 Scores per Class
        f1_per_class = f1_score(all_labels[:, i], all_preds[:, i], average=None)
        plt.figure(figsize=(8, 4))
        plt.bar(x, f1_per_class, width=0.6)
        plt.axhline(y=f1, color='r', linestyle='-', label=f'Macro F1: {f1:.3f}')
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.title(f"{metric} - F1 Score per Class")
        plt.xticks(x, x)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_f1_scores.png", dpi=300)
        plt.show()
    
    # Print summary of results
    print("\n--- Performance Summary ---")
    avg_acc = np.mean([results[m]['accuracy'] for m in metrics])
    avg_f1 = np.mean([results[m]['f1'] for m in metrics])
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    for metric in metrics:
        print(f"{metric}: Acc={results[metric]['accuracy']:.4f}, F1={results[metric]['f1']:.4f}")
    
    # Class collapse analysis
    print("\n--- Class Collapse Analysis ---")
    for i, metric in enumerate(metrics):
        pred_dist = np.bincount(all_preds[:, i], minlength=4) / len(all_preds)
        min_class_pct = np.min(pred_dist)
        print(f"{metric} prediction distribution: {pred_dist}")
        if min_class_pct < 0.05:
            print(f"WARNING: {metric} shows class collapse (min class = {min_class_pct:.4f})")
        else:
            print(f"OK: {metric} has good class balance (min class = {min_class_pct:.4f})")
    
    return results

# Main Execution
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  
    # Step 1: (Optional) Precompute and cache best frames
    train_csv = LABELS_DIR / "TrainLabels.csv"
    val_csv = LABELS_DIR / "ValidationLabels.csv"
    test_csv = LABELS_DIR / "TestLabels.csv"
    
    cache_file_train = CACHE_DIR / f"precomputed_{Path(train_csv).stem}_frame_30.pkl"
    if not cache_file_train.exists():
        print("Precomputing best frames for training data...")
        precompute_best_frames(train_csv, FRAMES_DIR, num_frames=30)
    
    cache_file_val = CACHE_DIR / f"precomputed_{Path(val_csv).stem}_frame_30.pkl"      
    if not cache_file_val.exists():
        print("Precomputing best frames for validation data...")      
        precompute_best_frames(val_csv, FRAMES_DIR, num_frames=30)
        
    cache_file_test = CACHE_DIR / f"precomputed_{Path(test_csv).stem}_frame_30.pkl"
    if not cache_file_test.exists():
        print("Precomputing best frames for test data...")
        precompute_best_frames(test_csv, FRAMES_DIR, num_frames=30)
    
    # # Step 2: Run Optuna tuning with early stopping
    # n_trials = 12
    # db_path = Path(r"C:/Users/abhis/Downloads/Documents/Learner Engagement Project/notebooks/mobilenettcn_balance_tuning.db")
    # db_path.parent.mkdir(parents=True, exist_ok=True)
    # try:
    #     conn = sqlite3.connect(db_path)
    #     print(f"Database created/connected successfully at: {db_path}")
    #     conn.close()
    # except Exception as e:
    #     print(f"Error: {e}")

    # print(f"Database location: {db_path}")
    # print(f"Writable: {os.access(db_path.parent, os.W_OK)}")

    # study = optuna.create_study(
    #     pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=10),
    #     direction="minimize",
    #     study_name="mobilev2_tcn_study",
    #     storage=f"sqlite:///{db_path}",
    #     load_if_exists=True
    # )

    # print("Starting Optuna hyperparameter tuning...")
    # completed_trials = len([t for t in study.trials if t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED}])
    # pbar = tqdm(total=n_trials, desc="Optuna Trials", unit="trial", dynamic_ncols=True, initial=completed_trials)
    
    # def update(study, trial):
    #     pbar.update()
    # study.optimize(objective, n_trials=n_trials, catch=(Exception,), callbacks=[update])
    # pbar.close()
    # print(f"Optuna tuning complete.\nBest trial: {study.best_trial}")
    
    # print("Starting Optuna hyperparameter tuning...")
    # print("\n--- Trial Status Summary ---")
    # completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    # failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    # print(f"Completed: {completed}, Failed: {failed}")

    # if completed == 0:
    #     print("\nERROR: No trials completed successfully!")
    #     print("Possible causes:")
    #     print("- All trials ran out of GPU memory")
    #     print("- Hyperparameter ranges too aggressive")
    #     print("- Bugs in trial objective function")
    #     exit(1)


    # Step 2: Get best hyperparameters from existing study
    print("Skipping hyperparameter tuning - using best parameters from existing study...")

    db_path = Path(r"C:/Users/abhis/Downloads/Documents/Learner Engagement Project/notebooks/mobilenettcn_balance_tuning.db")
    if not db_path.exists():
        print(f"Error: Database file {db_path} does not exist!")
        exit(1)

    # Clean up orphaned RUNNING trials
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'")
    conn.commit()
    print(f"Cleaned up orphaned RUNNING trials in the database")
    conn.close()

    # Load the study
    study = optuna.load_study(
        study_name="mobilev2_tcn_study",
        storage=f"sqlite:///{db_path}"
    )

    # Print study summary
    print("\n--- Trial Status Summary ---")
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    print(f"Completed: {completed}, Failed: {failed}")

    try:
        # Get best trial
        best_trial = study.best_trial
        print(f"Best trial: #{best_trial.number} with value: {best_trial.value}")
        params = best_trial.params
        print(f"Best parameters: {params}")
    except Exception as e:
        print(f"Error accessing best trial: {e}")
        # Fallback 
        print("Using fallback parameters from trial 11")
        params = {
            "num_frames": 30,
            "batch_size": 4,  
            "lr": 9.86649999152203e-05,
            "hidden_dim": 128,  
            "dropout_rate": 0.22722702432335967
        }
    # Step 3: Final training with best hyperparameters and early stopping
    # final_checkpoint = MODEL_DIR / "final_model_balance_checkpoint.pth" 
    final_checkpoint = MODEL_DIR / "final_model_balance_0.pth" #the above is the real file name, this a single model checkpoint
    
    if not final_checkpoint.exists():
        print("\n--- Starting Final Training ---")
        best_trial = study.best_trial
        params = best_trial.params

        num_frames = params.get("num_frames", 30)
        batch_size = params.get("batch_size", 8)
        hidden_dim = params.get("hidden_dim", 128)
        dropout_rate = params.get("dropout_rate", 0.25)

        train_dataset = VideoDataset(train_csv, FRAMES_DIR, 
                                    transform=train_transform, 
                                    num_frames=num_frames)
        val_dataset = VideoDataset(val_csv, FRAMES_DIR,
                                transform=val_transform,
                                num_frames=num_frames)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)

            #     # Create models for ensemble (3 models with different seeds)
            #     models = []
            #     histories = []
                
            #     for seed_idx in range(3):
            #         print(f"\n=== Training Model {seed_idx+1}/3 ===")
            #         torch.manual_seed(42 + seed_idx)
            #         torch.cuda.manual_seed(42 + seed_idx)
                    
            #         model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
            #         checkpoint_path = MODEL_DIR / f"final_model_balance_{seed_idx}.pth"
                    
            #         model, f1, history = train_model_with_phases(
            #             model, train_loader, val_loader, checkpoint_path, num_epochs=30
            #         )
                    
            #         models.append(model)
            #         histories.append(history)
            #         print(f"Model {seed_idx+1} best F1: {f1:.4f}")
                
            #     # Create ensemble
            #     ensemble = EnsembleModel(models)
            #     torch.save({
            #         'model_state_dict': ensemble.state_dict(),
            #         'models': [m.state_dict() for m in models],
            #         'params': params
            #     }, final_checkpoint)
            # else:
            #     print("\n--- Skipping Final Training (Checkpoint Exists) ---")
            #     print(f"Using existing model from: {final_checkpoint}")

    # print("\n--- Starting Evaluation ---")

    
    # params = best_trial.params
    # num_frames = params.get("num_frames", 30)
    # batch_size = params.get("batch_size", 8)
    # hidden_dim = params.get("hidden_dim", 128)
    # dropout_rate = params.get("dropout_rate", 0.25)

    # try:
    #     # Load ensemble if it exists
    #     checkpoint = torch.load(final_checkpoint, map_location=device)
        
    #     if 'models' in checkpoint:
    #         # This is an ensemble checkpoint
    #         individual_models = []
    #         for state_dict in checkpoint['models']:
    #             model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    #             model.load_state_dict(state_dict)
    #             model.to(device)
    #             individual_models.append(model)
            
    #         eval_model = EnsembleModel(individual_models)
    #     else:
    #         # Single model checkpoint
    #         eval_model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    #         eval_model.load_state_dict(checkpoint['model_state_dict'])
            
    #     eval_model.to(device)
    # except Exception as e:
    #     print(f"Error loading checkpoint: {e}")
    #     print("Creating a new model with the best parameters")
    #     eval_model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    #     eval_model.to(device)
    
    print("\n--- Starting Evaluation ---")

    # Make sure we have parameters for model creation
    params = best_trial.params
    num_frames = params.get("num_frames", 30)
    batch_size = params.get("batch_size", 4)
    hidden_dim = params.get("hidden_dim", 128)
    dropout_rate = params.get("dropout_rate", 0.25)

    # Path to single model checkpoint (not ensemble)
    single_model_checkpoint = MODEL_DIR / "final_model_balance_0.pth"

    try:
        print(f"Loading single model from: {single_model_checkpoint}")
        checkpoint = torch.load(single_model_checkpoint, map_location=device)
        
        # Create model with correct parameters
        eval_model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            eval_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Some checkpoints might store the state dict directly
            eval_model.load_state_dict(checkpoint)
            
        eval_model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating a new model with the best parameters instead")
        eval_model = EmotionSpecialistMobileTCN(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        eval_model.to(device)

    # Setup test dataset and dataloader
    test_dataset = VideoDataset(test_csv, FRAMES_DIR,
                            transform=val_transform,
                            num_frames=num_frames)

    test_loader = DataLoader(test_dataset,
                            batch_size=4,  # smaller batch size for evaluation
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    try:
        evaluate_model(eval_model, test_loader)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("\nERROR: Insufficient GPU memory for evaluation!")
            print("Try reducing batch size to 2 for evaluation")
            
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, 
                                num_workers=2, pin_memory=True)
            evaluate_model(eval_model, test_loader)
        else:
            raise