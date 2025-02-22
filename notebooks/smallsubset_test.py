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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights, mobilenet_v2, MobileNet_V2_Weights
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import logging
import sqlite3
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint  # For gradient checkpointing
import threading

logging.basicConfig(level=logging.INFO)

# ------------------------------
# CONSTANTS & HYPERPARAMETERS
# ------------------------------
GRADIENT_ACCUM_STEPS = 4  # Accumulate gradients over 4 mini-batches
NUM_FRAMES = 30           # Number of frames per video to use

# ------------------------------
# Environment & Paths
# ------------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"  # Expects folders like Train, Validation, Test
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

print("Models directory exists:", os.path.exists(MODEL_DIR))
print("Checkpoint path writable:", os.access(MODEL_DIR, os.W_OK))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Data Transforms
# ------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Helper Functions
# ------------------------------
def get_csv_clip_id(video_stem: str) -> str:
    base = video_stem.strip()
    return base.replace("110001", "202614", 1) if base.startswith("110001") else base

def select_impactful_frames(video_folder: Path, num_frames=30):
    # For testing we simply choose evenly spaced frames.
    frame_files = sorted(video_folder.glob("frame_*.jpg"))
    total = len(frame_files)
    if total == 0:
        return []
    if total <= num_frames:
        return frame_files
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [frame_files[i] for i in indices]

def precompute_best_frames(csv_file: Path, video_root: Path, num_frames=30):
    """
    Precompute and cache the frame file paths for each video.
    Saves a .pkl file with keys "valid_indices" and "precomputed_frames".
    The folder (e.g., 'Train') is derived from the CSV stem.
    """
    data = pd.read_csv(csv_file, dtype=str)
    data.columns = data.columns.str.strip()
    split = csv_file.stem.replace("Labels", "").replace("small_", "").strip()
    valid_indices = []
    precomputed = []
    skipped = 0
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Precomputing frames"):
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
    cache_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Precomputed results saved to {cache_file}")
    return cache_data

# To avoid pickling the LMDB Environment, we create a per-worker environment using thread-local storage.
_worker_env = threading.local()
def get_worker_env(lmdb_path):
    if not hasattr(_worker_env, 'env'):
        _worker_env.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    return _worker_env.env

def convert_pkl_to_lmdb(csv_file: Path, num_frames=30, transform=train_transform, lmdb_map_size=1 * 1024**3):
    """
    Convert the precomputed .pkl file into an LMDB database containing preprocessed
    feature tensors (half precision) from EfficientNet-B3.
    """
    pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}.pkl"
    lmdb_path = CACHE_DIR / f"lmdb_{csv_file.stem}_frame_{num_frames}"
    env = lmdb.open(str(lmdb_path), map_size=lmdb_map_size)
    
    with env.begin(write=False) as txn:
        if txn.stat()['entries'] > 0:
            print(f"LMDB database already exists at {lmdb_path}")
            env.close()
            return lmdb_path

    if not pkl_file.exists():
        precompute_best_frames(csv_file, FRAMES_DIR, num_frames=num_frames)
    with open(pkl_file, "rb") as f:
        cache = pickle.load(f)
    valid_indices = cache["valid_indices"]
    file_paths_list = cache["precomputed_frames"]

    # Prepare EfficientNet-B3 feature extractor (frozen)
    feature_extractor = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features
    feature_extractor.eval()
    feature_extractor.to(device)
    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("Converting file paths to LMDB preprocessed feature tensors...")
    with env.begin(write=True) as txn:
        for idx, paths in tqdm(enumerate(file_paths_list), total=len(file_paths_list)):
            video_features = []
            for fp in paths:
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    img = Image.new('RGB', (224, 224))
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    feat = feature_extractor(tensor)
                    feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).view(-1).cpu().half()
                video_features.append(feat)
            if video_features:
                video_features = torch.stack(video_features)
                # Use original CSV index from valid_indices for the key.
                key = f"video_{valid_indices[idx]}".encode("utf-8")
                txn.put(key, pickle.dumps(video_features))
    env.close()
    print(f"LMDB database created at {lmdb_path}")
    return lmdb_path

# ------------------------------
# LMDB Dataset Class (Corrected)
# ------------------------------
class VideoDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self, csv_file, lmdb_path, num_frames=30):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.data.columns = self.data.columns.str.strip()
        pkl_file = CACHE_DIR / f"precomputed_{csv_file.stem}_frame_{num_frames}.pkl"
        with open(pkl_file, "rb") as f:
            cache = pickle.load(f)
        # Store the original valid indices
        self.valid_indices = cache["valid_indices"]
        # Filter data using these valid indices and reset the index (for labels)
        self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)
        self.num_frames = num_frames
        self.lmdb_path = str(lmdb_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        env = get_worker_env(self.lmdb_path)
        original_idx = self.valid_indices[idx]  # Use original CSV index for key lookup
        key = f"video_{original_idx}".encode("utf-8")
        with env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise IndexError(f"Key {key} not found in LMDB")
            features = pickle.loads(data_bytes)
        labels = self.data.iloc[idx][["Engagement", "Boredom", "Confusion", "Frustration"]].astype(int)
        return features, torch.tensor(labels.values, dtype=torch.long)

# ------------------------------
# LSTM Model for Precomputed Features
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, feature_dim=1536, hidden_size=128, num_lstm_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4 * 4)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.view(-1, 4, 4)

# ------------------------------
# Training Function with Gradient Accumulation
# ------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, checkpoint_path, patience=5, gradient_accum_steps=GRADIENT_ACCUM_STEPS):
    model.to(device, non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
    else:
        start_epoch, best_val_loss = 0, float('inf')
    loss_fn = nn.CrossEntropyLoss().to(device)
    early_stop_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(features)
                outputs = outputs.view(outputs.size(0), 4, 4)
                loss = sum(loss_fn(outputs[:, d], labels[:, d]) for d in range(4)) / 4.0
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
                loss = sum(loss_fn(outputs[:, d], labels[:, d]) for d in range(4)) / 4.0
                val_loss += loss.item() * features.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                "epoch": epoch + 1,
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
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best_val_loss:.4f}")
            break
    return best_val_loss

# ------------------------------
# Hyperparameter Tuning with Optuna (using LMDB dataset)
# ------------------------------
def objective(trial):
    torch.cuda.empty_cache()
    gc.collect()
    num_frames = trial.suggest_categorical("num_frames", [NUM_FRAMES])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    epochs = trial.suggest_int("epochs", 3, 5)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    num_lstm_layers = trial.suggest_categorical("num_lstm_layers", [1, 2])
    
    train_set = VideoDatasetLMDB(LABELS_DIR / "TrainLabels.csv", 
                 CACHE_DIR / f"lmdb_{(LABELS_DIR / 'TrainLabels.csv').stem}_frame_{NUM_FRAMES}",
                 num_frames=num_frames)
    val_set = VideoDatasetLMDB(LABELS_DIR / "ValidationLabels.csv", 
                 CACHE_DIR / f"lmdb_{(LABELS_DIR / 'ValidationLabels.csv').stem}_frame_{NUM_FRAMES}",
                 num_frames=num_frames)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False, prefetch_factor=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=False, prefetch_factor=1)
    
    model = LSTMModel(feature_dim=1536, hidden_size=hidden_size, num_lstm_layers=num_lstm_layers).to(device)
    trial_checkpoint = MODEL_DIR / f"lmdb_trial_eff_{trial.number}_checkpoint.pth"
    trial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        best_loss = train_model(model, train_loader, val_loader, epochs, lr, trial_checkpoint,
                                patience=3, gradient_accum_steps=GRADIENT_ACCUM_STEPS)
    except Exception as e:
        if trial_checkpoint.exists():
            trial_checkpoint.unlink()
        print(f"Trial {trial.number} failed: {e}")
        best_loss = float("inf")
    del model, train_loader, val_loader, train_set, val_set
    torch.cuda.empty_cache()
    gc.collect()
    return best_loss

# ------------------------------
# Evaluation Function (using LMDB dataset)
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(features)
            outputs = outputs.view(outputs.size(0), 4, 4)
            preds = torch.argmax(outputs, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    for i, dim in enumerate(["Engagement", "Boredom", "Confusion", "Frustration"]):
        print(f"Classification report for {dim}:")
        print(classification_report(all_labels[:, i], all_preds[:, i], digits=3))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {dim}")
        plt.colorbar()
        plt.xticks(np.arange(cm.shape[0]), np.arange(cm.shape[0]))
        plt.yticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

# ------------------------------
# Main Execution (Small Subset Test)
# ------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    import io  # needed for LMDB conversion
    # Create a small CSV from the full TrainLabels.csv (first 10 rows)
    full_train_csv = LABELS_DIR / "TrainLabels.csv"
    small_train_csv = CACHE_DIR / "small_TrainLabels.csv"
    if not small_train_csv.exists():
        df = pd.read_csv(full_train_csv, dtype=str)
        df_small = df.head(10)
        df_small.to_csv(small_train_csv, index=False)
        print("Created small CSV with 10 rows.")
    else:
        print("Small CSV already exists.")
    
    # For testing, use the small CSV for training; full CSVs for validation/test
    train_csv = small_train_csv
    val_csv = LABELS_DIR / "ValidationLabels.csv"
    test_csv = LABELS_DIR / "TestLabels.csv"
    
    # Precompute frame paths and convert to LMDB for the small CSV
    precompute_best_frames(train_csv, FRAMES_DIR, NUM_FRAMES)
    LMDB_TRAIN = convert_pkl_to_lmdb(train_csv, NUM_FRAMES, transform=train_transform, lmdb_map_size=1 * 1024**3)
    
    # Check small LMDB dataset length and sample retrieval
    train_dataset = VideoDatasetLMDB(train_csv, LMDB_TRAIN, num_frames=NUM_FRAMES)
    print("Small LMDB dataset length:", len(train_dataset))
    if len(train_dataset) > 0:
        features, labels = train_dataset[0]
        print("Sample features shape:", features.shape)
        print("Sample labels:", labels)
    else:
        print("No valid videos in the small subset.")
    
    # Quick test of the training loop on the small dataset (one mini-iteration)
    model = LSTMModel(feature_dim=1536, hidden_size=64, num_lstm_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    model.train()
    for features, labels in tqdm(loader, desc="Testing Training Loop"):
        features = features.to(device)
        labels = labels.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(features)
            outputs = outputs.view(outputs.size(0), 4, 4)
            loss = sum(nn.CrossEntropyLoss()(outputs[:, d], labels[:, d]) for d in range(4)) / 4.0
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        print("Loss:", loss.item())
        break  # only one iteration for testing
