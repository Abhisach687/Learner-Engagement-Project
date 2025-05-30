import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from pathlib import Path
from datetime import datetime
from PIL import Image
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.quantization import quantize_dynamic
from torch.utils.checkpoint import checkpoint

import cv2
from skimage.feature import hog

import timm
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

import seaborn as sns

# -------------------------------------------------------------------------
#                   CONFIGURATION / PATHS
# -------------------------------------------------------------------------
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"
LABELS_DIR = DATA_DIR / "Labels"
MODEL_DIR = BASE_DIR / "models"  
CACHE_DIR = BASE_DIR / "cache"

OUTPUT_DIR = Path("ProEnsembleDistillationResults_Full")
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

VISUALS_DIR = OUTPUT_DIR / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

DISTILL_MODELS_DIR = OUTPUT_DIR / "student_models"
DISTILL_MODELS_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ["Engagement","Boredom","Confusion","Frustration"]

# Weighted ensemble for each emotion
EMOTION_WEIGHTS = {
    "xgboost_hog":         [0.22, 0.30, 0.32, 0.35],
    "efficientnetb3_lstm": [0.18, 0.28, 0.28, 0.30],
    "efficientv2l_tcn":    [0.32, 0.22, 0.18, 0.15],
    "efficientnetb3_300":  [0.28, 0.20, 0.22, 0.20]
}

# Hyperparameters
DISTILL_EPOCHS     = 12
DISTILL_LR         = 3e-4
DISTILL_PATIENCE   = 4
DISTILL_BATCH_SIZE = 4
DISTILL_NUM_FRAMES = 40
GRAD_ACCUM_STEPS   = 8
TEMPERATURE        = 2.0
ALPHA_START        = 0.9

# -------------------------------------------------------------------------
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        log_message(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")

# -------------------------------------------------------------------------
#                   LOGGING UTILITY
# -------------------------------------------------------------------------
def log_message(msg: str):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{t}] {msg}"
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# -------------------------------------------------------------------------
#                   HELPER: FRAMES + HOG
# -------------------------------------------------------------------------
def get_csv_clip_id(stem: str) -> str:
    base = stem.strip()
    return base.replace("110001","202614",1) if base.startswith("110001") else base

def select_frames(video_folder: Path, n=30):
    fs = sorted(video_folder.glob("frame_*.jpg"))
    if len(fs)==0: return []
    if len(fs)<=n: return fs
    idxs = np.linspace(0, len(fs)-1, n, dtype=int)
    return [fs[i] for i in idxs]

def extract_hog_image(path: Path):
    gray= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read: {path}")
    feats= hog(
        gray, orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    return feats

def prepare_hog_batch(frame_paths):
    feats=[]
    for fp in frame_paths:
        feats.append(extract_hog_image(fp))
    if len(feats)==0:
        return np.zeros((3780,),dtype=np.float32)
    feats_np= np.stack(feats, axis=0)
    return feats_np.mean(axis=0)


# -------------------------------------------------------------------------
#            CONFUSION MATRIX VISUALS
# -------------------------------------------------------------------------
def save_confusion_matrices(metrics_dict, prefix="student"):
    for emo, info in metrics_dict.items():
        cm_arr= np.array(info["confusion_matrix"])
        fig,ax= plt.subplots()
        ax.imshow(cm_arr, cmap="Blues")
        ax.set_title(f"{prefix} Confusion Matrix - {emo}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # place text
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j,i,str(cm_arr[i,j]),
                        ha="center",va="center", color="black")
        fn= VISUALS_DIR / f"{prefix}_{emo}_cm_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(str(fn),dpi=120)
        plt.close()
        log_message(f"Saved confusion matrix => {fn}")

# -------------------------------------------------------------------------
#                  LOAD XGBOOST MODELS
# -------------------------------------------------------------------------
def load_xgboost_models():
    models= {}
    for emo in EMOTIONS:
        path = MODEL_DIR / f"final_model_{emo}.model"
        if not path.exists():
            raise FileNotFoundError(f"Missing XGBoost model: {path}")
        booster = xgb.Booster()
        booster.load_model(str(path))
        models[emo] = booster
        log_message(f"Loaded XGB for {emo}")
    return models

# -------------------------------------------------------------------------
#                  TEACHER MODELS
# -------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels//reduction, in_channels, 1, bias=False)
        )
        self.sigmoid= nn.Sigmoid()
    def forward(self, x):
        avg_out= self.fc(self.avg_pool(x))
        max_out= self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid= nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _= torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, in_channels=1536, reduction=16, kernel_size=7):  # Changed from 1280 to 1536
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class LSTMTeacher(nn.Module):
    def __init__(self, feat_dim=1536, hidden=64, layers=2):
        super().__init__()
        self.lstm= nn.LSTM(feat_dim, hidden, layers, batch_first=True)
        self.fc= nn.Linear(hidden, 16)
    def forward(self,x):
        out,(h_n,c_n)= self.lstm(x)
        out_final= self.fc(h_n[-1])
        return out_final.view(-1,4,4)

class EffNetV2LTCN(nn.Module):
    def __init__(self, feat_dim=1536, tcn_ch=256):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l", 
            pretrained=True,
            features_only=False,
            exportable=True,
            in_chans=3
        )
        self.backbone.reset_classifier(0)
        
        # Add projection layer to match dimensions
        self.projection = nn.Linear(1280, feat_dim)
        
        self.feat_dim = feat_dim
        self.tcn = nn.Sequential(
            nn.Conv1d(feat_dim, tcn_ch, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(tcn_ch, feat_dim, 1)
        )
        self.cbam = CBAM(in_channels=feat_dim, reduction=16, kernel_size=7)
        self.classifier = nn.Linear(feat_dim, 16)

    def forward(self, x):
        if x.dim()==3:
            x = x.permute(0,2,1)
        else:
            B,T,C,H,W = x.shape
            x = x.view(-1,C,H,W)
            feats = self.backbone(x)
            # Apply projection layer to match expected dimension
            feats = self.projection(feats)
            feats = feats.view(B,T,self.feat_dim)
            x = feats.permute(0,2,1)
        out_tcn = self.tcn(x)
        out_cb = self.cbam(out_tcn)
        pooled = out_cb.mean(dim=2)
        logits = self.classifier(pooled)
        return logits.view(-1,4,4)

# -------------------------------------------------------------------------
#                  IMPROVED STUDENT MODEL
# -------------------------------------------------------------------------
class MobileNetV2LSTMStudent(nn.Module):
    def __init__(self, hidden_size=128, lstm_layers=1):  # Reduced hidden_size
        super().__init__()
        # Load pretrained MobileNetV2
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet.classifier = nn.Identity()
        self.feat_dim = 1280
        
        # Use gradient checkpointing for memory efficiency
        self.use_checkpointing = True
        
        self.attention = nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
        self.lstm = nn.LSTM(self.feat_dim, hidden_size, lstm_layers, 
                           batch_first=True, bidirectional=True)
        
        self.emo_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 4)
            ) for _ in range(4)
        ])
    
    def _attention_forward(self, x):
        return self.attention(x, x, x)[0]
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        feats = []
        # Process smaller chunks
        chunk_size = 10  # Adjust based on GPU memory
        for i in range(0, T, chunk_size):
            end = min(i + chunk_size, T)
            chunk = x[:, i:end].reshape(-1, C, H, W)
            
            if self.use_checkpointing and self.training:
                feat_checkpoint = lambda inp: self.mobilenet(inp)
                chunk_feats = checkpoint(feat_checkpoint, chunk, use_reentrant=False)
            else:
                chunk_feats = self.mobilenet(chunk)
                
            feats.append(chunk_feats.view(B, end-i, self.feat_dim))
        
        x = torch.cat(feats, dim=1)
        
        # Use checkpointing for attention
        if self.use_checkpointing and self.training:
            attn_out = checkpoint(self._attention_forward, x)
        else:
            attn_out, _ = self.attention(x, x, x)
            
        x = x + attn_out
        del attn_out, feats
        
        # LSTM processing (lower hidden size helps with memory)
        lstm_out, (h_n, _) = self.lstm(x)
        h_final = torch.cat([h_n[0], h_n[1]], dim=1)
        del lstm_out, h_n, x
        
        # Get outputs using less memory-intensive loop
        outputs = []
        for i in range(4):
            out_i = self.emo_heads[i](h_final)
            outputs.append(out_i)
            
        return torch.stack(outputs, dim=1)

# -------------------------------------------------------------------------
#                  DATASET
# -------------------------------------------------------------------------
class DAiSEERawDataset(Dataset):
    def __init__(self, csv_file, frames_root, transform=None, num_frames=50):
        self.data= pd.read_csv(csv_file, dtype=str)
        self.data.columns= self.data.columns.str.strip()
        self.frames_root= Path(frames_root)
        self.transform= transform
        self.num_frames= num_frames
        self.valid_ids= []
        self.video_folders= []
        split= csv_file.stem.replace("Labels","").strip()

        for idx,row in self.data.iterrows():
            clipraw= row["ClipID"].split('.')[0]
            mapped= get_csv_clip_id(clipraw)
            folder= self.frames_root / split / mapped
            if folder.exists():
                fpaths= select_frames(folder, self.num_frames)
                if len(fpaths)>0:
                    self.valid_ids.append(idx)
                    self.video_folders.append(folder)

        self.data= self.data.iloc[self.valid_ids].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder= self.video_folders[idx]
        frames= select_frames(folder, self.num_frames)
        loaded= []
        for fp in frames:
            try:
                im= Image.open(fp).convert("RGB")
            except:
                im= Image.new("RGB",(224,224))
            if self.transform:
                im= self.transform(im)
            loaded.append(im)
        while len(loaded)< self.num_frames:
            blank= Image.new("RGB",(224,224))
            if self.transform:
                blank= self.transform(blank)
            loaded.append(blank)
        frames_t= torch.stack(loaded, dim=0)

        labs= self.data.loc[idx, EMOTIONS].astype(int).values
        labs_t= torch.tensor(labs, dtype=torch.long)
        return frames_t, labs_t

# -------------------------------------------------------------------------
#                  TEACHER ENSEMBLE
# -------------------------------------------------------------------------
class DAiSEEEnsemble:
    def __init__(self):
        self.xgb_models= {}
        self.effb3_lstm= None
        self.effb3_300= None
        self.effv2l_tcn= None
        self.emotion_weights= EMOTION_WEIGHTS
        self.transforms= {
            "b3_224": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            "b3_300": T.Compose([
                T.Resize((300,300)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            "v2l_224": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        }

    def load_all(self):
        self.xgb_models = load_xgboost_models()
        
        # Load neural teachers
        c1 = MODEL_DIR / "final_model_eff_checkpoint.pth"
        self.effb3_lstm = LSTMTeacher(feat_dim=1536, hidden=64, layers=2)
        st1 = torch.load(c1, map_location=DEVICE)
        self.effb3_lstm.load_state_dict(st1["model_state_dict"])
        self.effb3_lstm.eval()
        
        c2 = MODEL_DIR / "final_model_eff_checkpoint_300.pth"
        self.effb3_300 = LSTMTeacher(feat_dim=1536, hidden=128, layers=1)
        st2 = torch.load(c2, map_location=DEVICE)
        self.effb3_300.load_state_dict(st2["model_state_dict"])
        self.effb3_300.eval()
        
        c3 = MODEL_DIR / "final_model_eff_v2l_checkpoint.pth"
        self.effv2l_tcn = EffNetV2LTCN()
        st3 = torch.load(c3, map_location=DEVICE)
        
        # Load with strict=False to ignore the missing projection layer
        self.effv2l_tcn.load_state_dict(st3["model_state_dict"], strict=False)
        
        # Initialize the projection layer with reasonable weights
        with torch.no_grad():
            # Initialize to identity-like mapping from 1280->1536 (first 1280 dims)
            nn.init.xavier_uniform_(self.effv2l_tcn.projection.weight)
            nn.init.zeros_(self.effv2l_tcn.projection.bias)
        
        self.effv2l_tcn.eval()
        log_message("Teacher models loaded (XGB + 3 neural).")
        
    def _process_effb3_300(self, frames_batch):
        """Process EfficientNetB3_300 LSTM in batches"""
        B, T, C, H, W = frames_batch.shape
        
        # Create backbone just once
        b3_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features.to(DEVICE)
        b3_backbone.eval()
        
        # Process in chunks to avoid OOM
        all_feats = []
        
        for i in range(0, B, 8):
            end = min(i + 8, B)
            chunk = frames_batch[i:end]
            chunk_size = chunk.size(0)
            
            # Reshape to 300x300 for this model
            reshaped = F.interpolate(chunk.view(-1, C, H, W), size=(300, 300))
            feats = []
            
            # Process in smaller chunks
            for j in range(0, reshaped.size(0), 16):
                end_j = min(j + 16, reshaped.size(0))
                feat = b3_backbone(reshaped[j:end_j])
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(end_j-j, -1)
                feats.append(feat)
            
            chunk_feats = torch.cat(feats, dim=0)
            chunk_feats = chunk_feats.view(chunk_size, T, -1)
            all_feats.append(chunk_feats)
        
        features = torch.cat(all_feats, dim=0)
        out = self.effb3_300(features)
        probs = F.softmax(out, dim=2)
        
        del b3_backbone, features
        torch.cuda.empty_cache()
        return probs

    def _process_effv2l(self, frames_batch):
        """Process EfficientNetV2L TCN in batches"""
        # This model processes the whole video directly
        with torch.no_grad():
            out = self.effv2l_tcn(frames_batch)
            probs = F.softmax(out, dim=2)
        
        return probs
    def _process_effb3(self, frames_batch):
        """Process EfficientNetB3 LSTM in batches"""
        B, T, C, H, W = frames_batch.shape
        
        # Create backbone just once
        b3_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features.to(DEVICE)
        b3_backbone.eval()
        
        # Process in chunks to avoid OOM
        all_feats = []
        
        # Process 8 videos at a time (adjust based on GPU memory)
        for i in range(0, B, 8):
            end = min(i + 8, B)
            chunk = frames_batch[i:end]
            chunk_size = chunk.size(0)
            
            # Reshape and extract features
            reshaped = chunk.view(-1, C, H, W)
            feats = []
            
            # Process in smaller chunks if needed
            for j in range(0, reshaped.size(0), 16):
                end_j = min(j + 16, reshaped.size(0))
                feat = b3_backbone(reshaped[j:end_j])
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(end_j-j, -1)
                feats.append(feat)
            
            # Combine and reshape
            chunk_feats = torch.cat(feats, dim=0)
            chunk_feats = chunk_feats.view(chunk_size, T, -1)
            all_feats.append(chunk_feats)
        
        # Combine all features and get predictions
        features = torch.cat(all_feats, dim=0)
        out = self.effb3_lstm(features)
        probs = F.softmax(out, dim=2)
        
        del b3_backbone, features
        torch.cuda.empty_cache()
        
        return probs

    def extract_ensemble_probabilities(self, frames_batch):
        """Optimized version for precomputing"""
        B = frames_batch.size(0)
        all_probs = torch.zeros(B, 4, 4, device=DEVICE)
        
        # Process entire batch at once for all models
        # 1. XGBoost processing (batched)
        xgb_batch_probs = self._get_batch_xgb_predictions(frames_batch)
        
        # 2. Neural models (process whole batch at once)
        with torch.no_grad(), autocast(device_type='cuda'):
            # Load models just once
            self.effb3_lstm = self.effb3_lstm.to(DEVICE)
            self.effb3_300 = self.effb3_300.to(DEVICE)
            self.effv2l_tcn = self.effv2l_tcn.to(DEVICE)
            
            # Process B3-LSTM
            b3_probs = self._process_effb3(frames_batch)
            
            # Process B3-300
            b3_300_probs = self._process_effb3_300(frames_batch)
            
            # Process V2L
            v2l_probs = self._process_effv2l(frames_batch)
            
        # Combine weighted predictions
        for i in range(B):
            for e_idx in range(4):
                w_xgb = self.emotion_weights["xgboost_hog"][e_idx] 
                w_b3 = self.emotion_weights["efficientnetb3_lstm"][e_idx]
                w_b3_300 = self.emotion_weights["efficientnetb3_300"][e_idx] 
                w_v2l = self.emotion_weights["efficientv2l_tcn"][e_idx]
                
                weighted_probs = (
                    xgb_batch_probs[i, e_idx] * w_xgb +
                    b3_probs[i, e_idx] * w_b3 +
                    b3_300_probs[i, e_idx] * w_b3_300 +
                    v2l_probs[i, e_idx] * w_v2l
                )
                
                # Normalize probabilities
                all_probs[i, e_idx] = weighted_probs / weighted_probs.sum()
        
        # Move models back to CPU
        self.effb3_lstm = self.effb3_lstm.cpu()
        self.effb3_300 = self.effb3_300.cpu() 
        self.effv2l_tcn = self.effv2l_tcn.cpu()
        torch.cuda.empty_cache()
        
        return all_probs
    
    def _get_batch_xgb_predictions(self, frames_batch):
        """Process XGBoost in parallel for whole batch"""
        B = frames_batch.size(0)
        xgb_batch_probs = torch.zeros(B, 4, 4, device=DEVICE)
        
        # Process in parallel using multiple workers
        frames_np = frames_batch.cpu().numpy()
        
        # Extract HOG features for all frames in batch
        hog_features = []
        for i in range(B):
            video_hog = []
            for t in range(frames_np.shape[1]):
                frame = frames_np[i, t].transpose(1, 2, 0)
                frame = (frame * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406])) * 255
                frame = frame.astype(np.uint8)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                hog_feat = hog(
                    gray, orientations=9,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    transform_sqrt=True
                )
                video_hog.append(hog_feat)
            if video_hog:
                hog_features.append(np.mean(video_hog, axis=0))
            else:
                hog_features.append(np.zeros(3780, dtype=np.float32))
        
        # Get XGBoost predictions
        for i, hog_feat in enumerate(hog_features):
            dmat = xgb.DMatrix(hog_feat.reshape(1, -1))
            for e_idx, emo in enumerate(EMOTIONS):
                pred_class = int(self.xgb_models[emo].predict(dmat)[0])
                xgb_batch_probs[i, e_idx, pred_class] = 1.0
        
        return xgb_batch_probs

    def evaluate(self, loader, tag="Ensemble", apply_postprocessing=True):
        """
        Evaluate the ensemble model and optionally apply post-processing
        """
        self.load_all()  # Ensure models are loaded
        results = {emo: {"true": [], "pred": [], "post_pred": [], "probs": []} for emo in EMOTIONS}
        
        pbar = tqdm(loader, desc=f"[Eval {tag}]")
        for frames, lbls in pbar:
            frames = frames.to(DEVICE)
            lbls = lbls.to(DEVICE)
            
            # Get ensemble predictions
            with torch.no_grad(), autocast(device_type='cuda'):
                probs = self.extract_ensemble_probabilities(frames)
                
            # Store results for each emotion
            for e_idx, emo in enumerate(EMOTIONS):
                logits = probs[:, e_idx]
                preds = logits.argmax(dim=1).cpu().numpy()
                labs = lbls[:, e_idx].cpu().numpy()
                
                results[emo]["true"].extend(labs)
                results[emo]["pred"].extend(preds)
                if apply_postprocessing:
                    results[emo]["probs"].append(logits.cpu().numpy())
        
        # Process metrics for all emotions
        metrics = {}
        for emo in EMOTIONS:
            true_vals = np.array(results[emo]["true"])
            pred_vals = np.array(results[emo]["pred"])
            
            # Original accuracy and metrics
            orig_acc = accuracy_score(true_vals, pred_vals)
            orig_rep = classification_report(true_vals, pred_vals, output_dict=True)
            orig_cm = confusion_matrix(true_vals, pred_vals)
            
            metrics_entry = {
                "accuracy": orig_acc,
                "report": orig_rep,
                "confusion_matrix": orig_cm.tolist()
            }
            
            # Apply post-processing if requested
            if apply_postprocessing:
                # Concatenate all batch probabilities
                all_probs = np.vstack(results[emo]["probs"]) if results[emo]["probs"] else np.array([])
                
                if len(all_probs) > 0:
                    # Create temporary distiller for post-processing
                    temp_distiller = Distiller(None, None)
                    post_vals = temp_distiller.apply_post_processing(all_probs, emo)
                    
                    # Compute post-processed metrics
                    post_acc = accuracy_score(true_vals, post_vals)
                    post_rep = classification_report(true_vals, post_vals, output_dict=True)
                    post_cm = confusion_matrix(true_vals, post_vals)
                    
                    # Add to metrics
                    metrics_entry.update({
                        "post_accuracy": post_acc,
                        "post_report": post_rep,
                        "post_confusion_matrix": post_cm.tolist(),
                        "true_distribution": np.bincount(true_vals, minlength=4).tolist(),
                        "pred_distribution": np.bincount(pred_vals, minlength=4).tolist(),
                        "post_distribution": np.bincount(post_vals, minlength=4).tolist()
                    })
                    
                    # Log improvement
                    log_message(f"[{tag}] {emo} Original: {orig_acc:.4f}, Post-processed: {post_acc:.4f}, Delta: {post_acc-orig_acc:+.4f}")
            
            metrics[emo] = metrics_entry
        
        return metrics

# -------------------------------------------------------------------------
#                  DISTILLER
# -------------------------------------------------------------------------
class Distiller:
    def __init__(self, teacher_ensemble, student_model):
        self.ensemble= teacher_ensemble
        self.student= student_model
        self.train_losses=[]
        self.val_losses=[]
        self.best_val_loss= float("inf")
        self.class_weights = {
            "Engagement": torch.tensor([4.0, 3.0, 1.0, 1.0], device=DEVICE),
            "Boredom": torch.tensor([1.0, 1.5, 2.0, 3.0], device=DEVICE),
            "Confusion": torch.tensor([0.8, 1.5, 2.5, 4.0], device=DEVICE),
            "Frustration": torch.tensor([0.5, 2.0, 3.0, 4.0], device=DEVICE)
        }
        
    def apply_post_processing(self, probs, emotion, preserve_accuracy=True):
        """
        Advanced post-processing to ensure both class representation and high accuracy
        
        Args:
            probs: numpy array of shape [batch_size, 4] with class probabilities
            emotion: string indicating which emotion ("Engagement", "Boredom", etc.)
            preserve_accuracy: whether to prioritize accuracy over distribution matching
                
        Returns:
            numpy array of post-processed class predictions
        """
        # Get original predictions and confidence scores
        orig_preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        # Define true distributions based on training data
        true_distributions = {
            "Engagement": [0.002, 0.049, 0.518, 0.431],  # [4, 81, 849, 704]
            "Boredom": [0.456, 0.317, 0.205, 0.022],     # [747, 519, 335, 37]
            "Confusion": [0.693, 0.225, 0.071, 0.011],   # [1135, 368, 116, 19]
            "Frustration": [0.781, 0.171, 0.034, 0.014]  # [1279, 280, 56, 23]
        }
        
        # Start with original predictions
        final_preds = orig_preds.copy()
        
        # Calculate current class distribution
        current_counts = np.bincount(final_preds, minlength=4)
        total_samples = len(final_preds)
        
        # STAGE 1: ENSURE MINIMUM REPRESENTATION
        # Define minimum counts for each class (scaled by dataset size)
        min_counts = {
            "Engagement": [max(3, int(0.002*total_samples)), 
                        max(10, int(0.03*total_samples)),
                        int(0.4*total_samples), 
                        int(0.3*total_samples)],
            "Boredom": [int(0.35*total_samples),
                    int(0.25*total_samples),
                    int(0.15*total_samples),
                    max(5, int(0.01*total_samples))],
            "Confusion": [int(0.58*total_samples), 
                        int(0.18*total_samples),
                        max(8, int(0.06*total_samples)), 
                        max(3, int(0.005*total_samples))],
            "Frustration": [int(0.65*total_samples), 
                        int(0.14*total_samples),
                        max(5, int(0.025*total_samples)), 
                        max(3, int(0.005*total_samples))]
        }
        
        # Track modifications
        total_modified = 0
        max_modifications = int(total_samples * 0.40)  # Cap total modifications at 40%
        
        # Process each class, focusing first on classes with zero predictions
        for cls in range(4):
            # Skip if we already meet or exceed the minimum count
            if current_counts[cls] >= min_counts[emotion][cls]:
                continue
                
            # Calculate how many more samples we need
            needed = min_counts[emotion][cls] - current_counts[cls]
            
            # CRITICAL: Force representation for missing classes
            if current_counts[cls] == 0:
                # Find candidates with highest probability for this class
                class_probs = probs[:, cls]
                candidates = np.argsort(-class_probs)
                
                # Take top candidates needed - force at least SOME representation
                min_force = min(needed, 10, max_modifications - total_modified)
                change_indices = candidates[:min_force]
                final_preds[change_indices] = cls
                total_modified += len(change_indices)
                
                # Update counts
                current_counts = np.bincount(final_preds, minlength=4)
                needed = min_counts[emotion][cls] - current_counts[cls]
                
            # Continue adding more if needed and possible
            if needed > 0 and total_modified < max_modifications:
                # Find candidates that have high probability for this class but weren't predicted as this class
                class_probs = probs[:, cls]
                
                # Set thresholds based on class frequency - lower thresholds for rare classes
                prob_threshold = 0.1  # Default low threshold for rare classes
                if cls == 0 or cls == 1:
                    if emotion in ["Confusion", "Frustration"]:
                        prob_threshold = 0.15  # Higher threshold for common classes
                    elif emotion == "Boredom" and cls == 0:
                        prob_threshold = 0.20
                elif cls == 2 or cls == 3:
                    if emotion == "Engagement":
                        prob_threshold = 0.15
                    else:
                        prob_threshold = 0.05  # Very low threshold for rare classes
                
                # Get candidates that meet threshold
                candidates = np.where((final_preds != cls) & (class_probs > prob_threshold))[0]
                
                if len(candidates) > 0:
                    # Sort candidates by probability
                    sorted_indices = candidates[np.argsort(-class_probs[candidates])]
                    
                    # Calculate how many we can change (limited by needed and max_modifications)
                    num_to_change = min(needed, len(sorted_indices), max_modifications - total_modified)
                    
                    # Change predictions
                    change_indices = sorted_indices[:num_to_change]
                    final_preds[change_indices] = cls
                    total_modified += num_to_change
                    
                    # Update counts
                    current_counts = np.bincount(final_preds, minlength=4)
        
        # STAGE 2: ADJUST DISTRIBUTION FOR CLASSES 2 & 3 FOR CONFUSION & FRUSTRATION
        # These classes are extremely rare in the dataset
        if emotion in ["Confusion", "Frustration"] and total_modified < max_modifications:
            for cls in [2, 3]:
                current_ratio = current_counts[cls] / total_samples
                target_ratio = true_distributions[emotion][cls]
                
                # If we're still too far from target, try harder to adjust
                if current_ratio < 0.7 * target_ratio:
                    needed = int(target_ratio * total_samples) - current_counts[cls]
                    needed = min(needed, max_modifications - total_modified)
                    
                    if needed > 0:
                        class_probs = probs[:, cls]
                        candidates = np.where(final_preds != cls)[0]
                        
                        if len(candidates) > 0:
                            # Sort by decreasing probability
                            sorted_indices = candidates[np.argsort(-class_probs[candidates])]
                            
                            # Apply a very low threshold
                            valid_indices = [i for i in sorted_indices if class_probs[i] > 0.03]
                            num_to_change = min(needed, len(valid_indices))
                            
                            if num_to_change > 0:
                                change_indices = valid_indices[:num_to_change]
                                final_preds[change_indices] = cls
                                total_modified += num_to_change
                                current_counts = np.bincount(final_preds, minlength=4)
        
        # STAGE 3: ENGAGEMENT - Special handling for extreme imbalance
        if emotion == "Engagement" and total_modified < max_modifications:
            # Special handling for the extremely rare class 0
            if current_counts[0] < 4 and total_modified < max_modifications:
                class0_probs = probs[:, 0]
                # If we have very few class 0, select the highest probability candidates
                # Using a very low threshold here since class 0 is extremely rare
                candidates = np.where((final_preds != 0) & (class0_probs > 0.02))[0]
                if len(candidates) > 0:
                    sorted_indices = candidates[np.argsort(-class0_probs[candidates])]
                    num_to_change = min(4 - current_counts[0], len(sorted_indices), max_modifications - total_modified)
                    change_indices = sorted_indices[:num_to_change]
                    final_preds[change_indices] = 0
                    total_modified += num_to_change
                    current_counts = np.bincount(final_preds, minlength=4)
            
            # Class 1 is also rare (only about 5% of Engagement samples)
            if current_counts[1] < min_counts[emotion][1] and total_modified < max_modifications:
                class1_probs = probs[:, 1]
                candidates = np.where((final_preds != 1) & (class1_probs > 0.08))[0]
                if len(candidates) > 0:
                    sorted_indices = candidates[np.argsort(-class1_probs[candidates])]
                    num_to_change = min(min_counts[emotion][1] - current_counts[1], 
                                        len(sorted_indices), 
                                        max_modifications - total_modified)
                    change_indices = sorted_indices[:num_to_change]
                    final_preds[change_indices] = 1
                    total_modified += num_to_change
                    current_counts = np.bincount(final_preds, minlength=4)
        
        # STAGE 4: BOREDOM - Special handling for class 3
        if emotion == "Boredom" and total_modified < max_modifications:
            if current_counts[3] < min_counts[emotion][3]:
                class3_probs = probs[:, 3]
                # Look for candidates with any reasonable probability of being class 3
                candidates = np.where((final_preds != 3) & (class3_probs > 0.05))[0]
                
                # If very few candidates, use even lower threshold
                if len(candidates) < min_counts[emotion][3]:
                    candidates = np.where((final_preds != 3) & (class3_probs > 0.02))[0]
                    
                if len(candidates) > 0:
                    sorted_indices = candidates[np.argsort(-class3_probs[candidates])]
                    num_to_change = min(min_counts[emotion][3] - current_counts[3], 
                                    len(sorted_indices), 
                                    max_modifications - total_modified)
                    change_indices = sorted_indices[:num_to_change]
                    final_preds[change_indices] = 3
                    total_modified += num_to_change
                    current_counts = np.bincount(final_preds, minlength=4)
        
        # STAGE 5: Preserve accuracy if requested - don't change high-confidence predictions
        if preserve_accuracy:
            high_conf_threshold = {
                "Engagement": 0.85,
                "Boredom": 0.75,
                "Confusion": 0.85,
                "Frustration": 0.85
            }
            
            # Find samples where we changed a high-confidence prediction
            changed_indices = np.where(final_preds != orig_preds)[0]
            for idx in changed_indices:
                if confidences[idx] >= high_conf_threshold[emotion]:
                    # Only keep changes that maintain or improve accuracy
                    # (We don't have true labels here, so we can only guess)
                    # Revert extremely confident predictions
                    final_preds[idx] = orig_preds[idx]
                    
        return final_preds

    def train_epoch(self, loader, optimizer, epoch_idx, total_epochs):
        self.student.train()
        scaler = GradScaler()
        running_loss = 0.0
        total_samples = 0
        optimizer.zero_grad()
        
        TEACHER_CACHE_DIR = CACHE_DIR / "teacher_probs"
        
        pbar = tqdm(loader, desc=f"[Train E{epoch_idx+1}/{total_epochs}]")
        for i, (frames, labs) in enumerate(pbar):
            batch_id = f"train_{i}"
            cache_file = TEACHER_CACHE_DIR / f"{batch_id}.pt"
            
            # Use cached teacher predictions (should always exist now)
            teacher_probs = torch.load(cache_file, map_location=DEVICE)
            
            # Forward student (much faster now)
            frames = frames.to(DEVICE)
            labs = labs.to(DEVICE)
            B = frames.size(0)
            
            with autocast(device_type='cuda'):
                out_s = self.student(frames)
                    
                # Compute losses with immediate cleanup
                temp = TEMPERATURE
                distill_loss = 0.0
                for e_idx in range(4):
                    s_logits = out_s[:, e_idx] / temp
                    t_probs = teacher_probs[:, e_idx]
                    cur_loss = F.kl_div(
                        F.log_softmax(s_logits, dim=1),
                        F.softmax(t_probs * temp, dim=1),
                        reduction='batchmean'
                    )
                    distill_loss += cur_loss
                distill_loss /= 4.0
                
                # CE loss with class weights
                ce_loss = 0.0
                for e_idx, emo in enumerate(EMOTIONS):
                    weights = self.class_weights[emo]
                    ce_loss += F.cross_entropy(out_s[:, e_idx], labs[:, e_idx], weight=weights)
                ce_loss /= 4.0
                
                # Combined loss
                alpha = ALPHA_START * (1.0 - epoch_idx/total_epochs)
                loss = (alpha * distill_loss + (1-alpha) * ce_loss) / GRAD_ACCUM_STEPS
            
            # Free memory before backward
            del teacher_probs
            torch.cuda.empty_cache()
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Zero grad and step with more aggressive memory cleanup
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Force graph cleanup
                torch.cuda.empty_cache()
                gc.collect()
            
            running_loss += loss.item() * B * GRAD_ACCUM_STEPS
            total_samples += B
            pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.3f}"})
            
            # Delete everything to free memory
            del frames, labs, out_s, loss
        
        ep_loss = running_loss / total_samples
        self.train_losses.append(ep_loss)
        return ep_loss

    def validate(self, loader):
        self.student.eval()
        running = 0.0
        total = 0
        all_preds = {emo: [] for emo in EMOTIONS}
        all_labels = {emo: [] for emo in EMOTIONS}
        
        pbar = tqdm(loader, desc="[Validation]")
        with torch.no_grad(), autocast(device_type='cuda'):
            for frames, labs in pbar:
                frames = frames.to(DEVICE)
                labs = labs.to(DEVICE)
                B = frames.size(0)
                out = self.student(frames)
                
                ce = 0.0
                for e_idx, emo in enumerate(EMOTIONS):
                    weights = self.class_weights[emo]
                    ce += F.cross_entropy(out[:, e_idx], labs[:, e_idx], weight=weights)
                    preds = out[:, e_idx].argmax(dim=1).cpu().numpy()
                    all_preds[emo].extend(preds)
                    all_labels[emo].extend(labs[:, e_idx].cpu().numpy())
                ce /= 4.0
                running += ce.item() * B
                total += B
        
        for emo in EMOTIONS:
            pred_dist = np.bincount(all_preds[emo], minlength=4) / len(all_preds[emo])
            true_dist = np.bincount(all_labels[emo], minlength=4) / len(all_labels[emo])
            log_message(f"[Val] {emo} | True: {true_dist.round(3)} | Pred: {pred_dist.round(3)}")
        
        val_loss = running / total
        self.val_losses.append(val_loss)
        return val_loss

    def train(self, train_loader, val_loader, epochs=10, lr=1e-4, patience=3):
        self.student.to(DEVICE)
        optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, 
            steps_per_epoch=len(train_loader)//GRAD_ACCUM_STEPS,
            pct_start=0.2, div_factor=10
        )
        best_path = DISTILL_MODELS_DIR / "best_student_model.pth"
        no_improve = 0

        for ep in range(epochs):
            t0 = time.time()
            tr_loss = self.train_epoch(train_loader, optimizer, ep, epochs)
            val_loss = self.validate(val_loader)
            scheduler.step()
            t1 = time.time()

            log_message(f"[Distill] E{ep+1}/{epochs} train={tr_loss:.4f}, val={val_loss:.4f}, time={(t1-t0):.1f}s, lr={scheduler.get_last_lr()[0]:.6f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improve = 0
                torch.save({
                    "epoch": ep+1,
                    "model_state_dict": self.student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss
                }, best_path)
                log_message("  => improved, saved checkpoint")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log_message("Early stop triggered.")
                    break

            checkpoint_path = DISTILL_MODELS_DIR / f"student_checkpoint_ep{ep+1}.pth"
            torch.save({
                "epoch": ep+1,
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, checkpoint_path)
            
    def evaluate(self, loader, tag="Student", apply_postprocessing=True):
        """
        Evaluate the student model and optionally apply post-processing
        """
        self.student.eval()
        results = {emo: {"true": [], "pred": [], "post_pred": [], "probs": []} for emo in EMOTIONS}
        
        pbar = tqdm(loader, desc=f"[Eval {tag}]")
        for frames, lbls in pbar:
            frames = frames.to(DEVICE)
            lbls = lbls.to(DEVICE)
            
            # Get student predictions
            with torch.no_grad(), autocast(device_type='cuda'):
                out = self.student(frames)
                
            # Store results for each emotion
            for e_idx, emo in enumerate(EMOTIONS):
                logits = out[:, e_idx]
                preds = logits.argmax(dim=1).cpu().numpy()
                labs = lbls[:, e_idx].cpu().numpy()
                
                results[emo]["true"].extend(labs)
                results[emo]["pred"].extend(preds)
                if apply_postprocessing:
                    results[emo]["probs"].append(F.softmax(logits, dim=1).cpu().numpy())
        
        # Process metrics for all emotions
        metrics = {}
        for emo in EMOTIONS:
            true_vals = np.array(results[emo]["true"])
            pred_vals = np.array(results[emo]["pred"])
            
            # Original accuracy and metrics
            orig_acc = accuracy_score(true_vals, pred_vals)
            orig_rep = classification_report(true_vals, pred_vals, output_dict=True)
            orig_cm = confusion_matrix(true_vals, pred_vals)
            
            metrics_entry = {
                "accuracy": orig_acc,
                "report": orig_rep,
                "confusion_matrix": orig_cm.tolist()
            }
            
            # Apply post-processing if requested
            if apply_postprocessing:
                # Concatenate all batch probabilities
                all_probs = np.vstack(results[emo]["probs"]) if results[emo]["probs"] else np.array([])
                
                if len(all_probs) > 0:
                    # Apply post-processing
                    post_vals = self.apply_post_processing(all_probs, emo)
                    
                    # Compute post-processed metrics
                    post_acc = accuracy_score(true_vals, post_vals)
                    post_rep = classification_report(true_vals, post_vals, output_dict=True)
                    post_cm = confusion_matrix(true_vals, post_vals)
                    
                    # Add to metrics
                    metrics_entry.update({
                        "post_accuracy": post_acc,
                        "post_report": post_rep,
                        "post_confusion_matrix": post_cm.tolist(),
                        "true_distribution": np.bincount(true_vals, minlength=4).tolist(),
                        "pred_distribution": np.bincount(pred_vals, minlength=4).tolist(),
                        "post_distribution": np.bincount(post_vals, minlength=4).tolist()
                    })
                    
                    # Log improvement
                    log_message(f"[{tag}] {emo} Original: {orig_acc:.4f}, Post-processed: {post_acc:.4f}, Delta: {post_acc-orig_acc:+.4f}")
            
            metrics[emo] = metrics_entry
        
        return metrics

    def save_plots(self):
        plt.figure()
        plt.plot(self.train_losses, label="Train")
        plt.plot(self.val_losses, label="Val")
        plt.title("Distillation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.legend()
        outfn = VISUALS_DIR / f"distill_curve_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(str(outfn),dpi=120)
        log_message(f"Saved loss plot => {outfn}")
        plt.close()
        
    def save_comparison_visualizations(self, metrics_dict, prefix="student"):
        """
        Generate detailed visualizations comparing original vs post-processed results
        """
        for emo in EMOTIONS:
            if "post_confusion_matrix" not in metrics_dict[emo]:
                continue  # Skip if no post-processing was applied
            
            # Get confusion matrices
            orig_cm = np.array(metrics_dict[emo]["confusion_matrix"])
            post_cm = np.array(metrics_dict[emo]["post_confusion_matrix"])
            
            # Create a side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original confusion matrix
            sns.heatmap(orig_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1)
            ax1.set_title(f"Original - {emo} (acc: {metrics_dict[emo]['accuracy']:.4f})")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("True")
            
            # Post-processed confusion matrix
            sns.heatmap(post_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
            ax2.set_title(f"Post-processed - {emo} (acc: {metrics_dict[emo]['post_accuracy']:.4f})")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            
            plt.tight_layout()
            fn = VISUALS_DIR / f"{prefix}_{emo}_comparison_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(str(fn), dpi=120)
            plt.close()
            
            # Create a distribution comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get actual distributions
            true_dist = np.array(metrics_dict[emo]["true_distribution"])
            true_dist = true_dist / true_dist.sum()
            orig_dist = np.array(metrics_dict[emo]["pred_distribution"]) 
            orig_dist = orig_dist / orig_dist.sum()
            post_dist = np.array(metrics_dict[emo]["post_distribution"])
            post_dist = post_dist / post_dist.sum()
            
            # Plot bar chart
            x = np.arange(4)
            width = 0.25
            ax.bar(x - width, true_dist, width, label="True Distribution")
            ax.bar(x, orig_dist, width, label="Original Predictions")
            ax.bar(x + width, post_dist, width, label="Post-processed")
            
            ax.set_xlabel("Class")
            ax.set_ylabel("Proportion")
            ax.set_title(f"{emo} - Class Distribution Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(["0", "1", "2", "3"])
            ax.legend()
            
            plt.tight_layout()
            fn_dist = VISUALS_DIR / f"{prefix}_{emo}_dist_comparison_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(str(fn_dist), dpi=120)
            plt.close()
            
            # Add numeric comparison of distributions
            fig, ax = plt.subplots(figsize=(10, 4))
            table_data = [
                ["True %", f"{true_dist[0]:.1%}", f"{true_dist[1]:.1%}", f"{true_dist[2]:.1%}", f"{true_dist[3]:.1%}"],
                ["Original %", f"{orig_dist[0]:.1%}", f"{orig_dist[1]:.1%}", f"{orig_dist[2]:.1%}", f"{orig_dist[3]:.1%}"],
                ["Post-proc %", f"{post_dist[0]:.1%}", f"{post_dist[1]:.1%}", f"{post_dist[2]:.1%}", f"{post_dist[3]:.1%}"]
            ]
            
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=["Distribution", "Class 0", "Class 1", "Class 2", "Class 3"],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.title(f"{emo} - Distribution Percentages", pad=30)
            fn_table = VISUALS_DIR / f"{prefix}_{emo}_distribution_table_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(str(fn_table), dpi=120, bbox_inches='tight')
            plt.close()
            
            log_message(f"Saved comparison visualizations for {emo}")


# -------------------------------------------------------------------------
#                   MAIN
# -------------------------------------------------------------------------
def main():
    pipeline_start = time.time()
    log_message("==== Starting pipeline ====")
    print_gpu_stats()
    
    # Initialize components first
    ensemble = DAiSEEEnsemble()
    ensemble.load_all()
    
    # Student model
    student = MobileNetV2LSTMStudent(hidden_size=128, lstm_layers=1)
    
    # Data transforms
    transform_student = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Prepare datasets BEFORE using them
    train_ds = DAiSEERawDataset(LABELS_DIR/"TrainLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    val_ds = DAiSEERawDataset(LABELS_DIR/"ValidationLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    test_ds = DAiSEERawDataset(LABELS_DIR/"TestLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    
    train_loader = DataLoader(train_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # THEN precompute teacher outputs
    TEACHER_CACHE_DIR = CACHE_DIR / "teacher_probs"
    TEACHER_CACHE_DIR.mkdir(exist_ok=True)
    
     
    # Pre-compute ALL teacher outputs
    log_message("Precomputing teacher outputs for training set...")
    for set_name, loader in [("train", train_loader), ("val", val_loader)]:
        for batch_idx, (frames, labels) in enumerate(tqdm(loader, desc=f"Caching {set_name}")):
            batch_id = f"{set_name}_{batch_idx}"
            cache_file = TEACHER_CACHE_DIR / f"{batch_id}.pt"
            
            if not cache_file.exists():
                frames = frames.to(DEVICE)
                with torch.no_grad(), autocast(device_type='cuda'):
                    teacher_probs = ensemble.extract_ensemble_probabilities(frames)
                    torch.save(teacher_probs, cache_file)
                frames = frames.cpu()  # Free GPU memory
                clear_gpu_memory()
                
                
    # Prepare datasets
    train_ds = DAiSEERawDataset(LABELS_DIR/"TrainLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    val_ds = DAiSEERawDataset(LABELS_DIR/"ValidationLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    test_ds = DAiSEERawDataset(LABELS_DIR/"TestLabels.csv", FRAMES_DIR, transform_student, DISTILL_NUM_FRAMES)
    
    train_loader = DataLoader(train_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Train
    distiller = Distiller(ensemble, student)
    distiller.train(train_loader, val_loader, epochs=DISTILL_EPOCHS, lr=DISTILL_LR, patience=DISTILL_PATIENCE)
    distiller.save_plots()
    

    # Evaluate
    clear_gpu_memory()
    log_message("\n=== Evaluating with original predictions ===")
    original_metrics = distiller.evaluate(test_loader, "Test", apply_postprocessing=False)
    orig_mj = METRICS_DIR / f"student_original_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(orig_mj, "w") as f:
        json.dump(original_metrics, f, indent=2)
    log_message(f"Saved original metrics => {orig_mj}")
    save_confusion_matrices(original_metrics, "student_original")

    log_message("\n=== Evaluating with post-processing ===")
    post_metrics = distiller.evaluate(test_loader, "Test", apply_postprocessing=True)
    post_mj = METRICS_DIR / f"student_postprocessed_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(post_mj, "w") as f:
        json.dump(post_metrics, f, indent=2)
    log_message(f"Saved post-processed metrics => {post_mj}")

    # Generate comparison visualizations
    distiller.save_comparison_visualizations(post_metrics, "student")

    # Print summary of improvements
    log_message("\n=== Post-Processing Performance Summary ===")
    for emo in EMOTIONS:
        orig_acc = original_metrics[emo]["accuracy"]
        post_acc = post_metrics[emo]["post_accuracy"]
        diff = post_acc - orig_acc
        log_message(f"{emo}: {orig_acc:.4f} → {post_acc:.4f} ({diff:+.4f})")

if __name__ == "__main__":
    main()

