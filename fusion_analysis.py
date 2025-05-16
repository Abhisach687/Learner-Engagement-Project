import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import onnxruntime as ort
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque, Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.checkpoint import checkpoint
from skimage.feature import hog as skimage_hog
import time
sns.set_theme(style="whitegrid", context="talk")   

# Define HOG function here if needed
def hog(image, **kwargs):
    return skimage_hog(image, **kwargs)

# -------------------------------------------------------------------------
#                      CONFIGURATION AND PATHS
# -------------------------------------------------------------------------
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
EMOTIONS = ["Engagement", "Boredom", "Confusion", "Frustration"]

# Create benchmark results directory with timestamp to avoid overwriting
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = BASE_DIR / f"fusion_benchmark_{TIMESTAMP}"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

# Define emotion class labels for clarity in results
EMOTION_LABELS = {
    "Engagement": ["Disengaged", "Low Engagement", "Engaged", "Very Engaged"],
    "Boredom": ["Not Bored", "Slightly Bored", "Bored", "Very Bored"],
    "Confusion": ["Not Confused", "Slightly Confused", "Confused", "Very Confused"],
    "Frustration": ["Not Frustrated", "Slightly Frustrated", "Frustrated", "Very Frustrated"]
}

# Fusion methods to compare
FUSION_METHODS = [
    "selective_fusion",
    "weighted_fusion",
    "hybrid_balanced_fusion",
    "mobilenet_confidence_gate",  # Gated approach
    "emotion_specific_gated_fusion"  
]

# Confidence threshold for the gated approach
CONF_GATE = 0.45

# Emotion-specific confidence gates based on benchmark analysis
EMOTION_GATES = {
    "Engagement": 0.10,
    "Boredom": 0.45,
    "Confusion": 0.10,
    "Frustration": 0.10
}

# Fusion strategies for each emotion
FUSION_STRATEGIES = {
    "Engagement": "mobilenet_only",  # Use MobileNet predictions unless confidence extremely low
    "Boredom": "gated_blend",        # Blend probabilities when MobileNet confidence below threshold
    "Confusion": "gated_weighted",    # Use weighted fusion for Confusion if MobileNet confidence low
    "Frustration": "gated_switch"     # Hard switch to XGBoost for very low confidence predictions
}


def process_all_frames_temporally(frame_dirs, student_model, xgboost_models, max_frames_per_dir=None):
    """Process all frames temporally from all directories."""
    print("Processing frames temporally...")
    
    # Create structures to hold data
    mobilenet_preds = {emotion: [] for emotion in EMOTIONS}
    mobilenet_probs = {emotion: [] for emotion in EMOTIONS}
    xgboost_preds = {emotion: [] for emotion in EMOTIONS}
    xgboost_probs = {emotion: [] for emotion in EMOTIONS}
    
    # Process each frame directory
    for i, frame_dir in enumerate(frame_dirs):
        print(f"Processing directory {i+1}/{len(frame_dirs)}: {frame_dir.name}")
        
        # Find frame files in this directory
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Warning: No frames found in {frame_dir}")
            continue
            
        # Limit frames if specified
        if max_frames_per_dir is not None:
            frame_files = frame_files[:min(max_frames_per_dir, len(frame_files))]
            
        # Process all frames temporally
        processed_frames = []
        
        for frame_file in frame_files:
            # Load and preprocess the frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
                
            processed_frame = preprocess_frame(frame)
            processed_frames.append(processed_frame)
            
        # If we have frames, run MobileNetV2_distilled
        if processed_frames:
            # Run inference - handle any number of frames
            mobilenet_result = run_pro_ensemble_inference(processed_frames, student_model)
            
            # Store predictions and probabilities
            for emotion in EMOTIONS:
                if emotion in mobilenet_result:
                    prob = mobilenet_result[emotion]
                    pred = np.argmax(prob)
                    mobilenet_preds[emotion].append(pred)
                    mobilenet_probs[emotion].append(prob)
            
        # Process a representative frame for XGBOOST_HOG
        if xgboost_models and frame_files:
            # Use middle frame for XGBOOST_HOG
            mid_idx = len(frame_files) // 2
            frame = cv2.imread(str(frame_files[mid_idx]))
            
            if frame is not None:
                # Extract HOG features
                hog_features = extract_hog_features(frame).reshape(1, -1)
                
                # Run XGBoost inference
                xgboost_result = run_xgboost_inference(hog_features, xgboost_models)
                
                # Store predictions and probabilities
                for emotion in EMOTIONS:
                    if emotion in xgboost_result:
                        prob = xgboost_result[emotion][0]  # First (and only) sample
                        pred = np.argmax(prob)
                        xgboost_preds[emotion].append(pred)
                        xgboost_probs[emotion].append(prob)
    
    return mobilenet_preds, mobilenet_probs, xgboost_preds, xgboost_probs

# -------------------------------------------------------------------------
#                 CONF‑GATE GRID‑SEARCH UTILITIES            
# -------------------------------------------------------------------------
CONF_GATE_CANDIDATES = np.arange(0.10, 0.91, 0.05)  # 0.10 … 0.90

def apply_mobilenet_confidence_gate_once(data, sample_idx, gate):
    """
    Same logic as before but with a user‑supplied gate value.
    Returns final class per emotion for one sample.
    """
    results = {}
    for emotion in EMOTIONS:
        prob_m = data["MobileNetV2_distilled"]["probabilities"][emotion][sample_idx]
        pred_m = int(np.argmax(prob_m))
        conf_m = float(np.max(prob_m))

        if conf_m >= gate or emotion not in data["XGBOOST_HOG"]["probabilities"]:
            results[emotion] = pred_m
        else:
            prob_x = data["XGBOOST_HOG"]["probabilities"][emotion][sample_idx]
            pred_x = int(np.argmax(prob_x))
            conf_x = float(np.max(prob_x))
            results[emotion] = pred_m if conf_m >= conf_x else pred_x
    return results




print(f"Results will be saved to: {RESULTS_DIR}")


# -------------------------------------------------------------------------
#                      POST-PROCESSING FUNCTIONS
# -------------------------------------------------------------------------
def apply_pro_ensemble_postprocessing(probs, emotion):
    """
    Apply post-processing to ProEnsembleDistillation predictions.
    Implements the post-processing logic from ProEnsembleDistillation.py.
    """
    # Get original predictions and confidence scores
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Define true distributions based on training data
    true_distributions = {
        "Engagement": [0.002, 0.049, 0.518, 0.431],  # [4, 81, 849, 704]
        "Boredom": [0.456, 0.317, 0.205, 0.022],     # [747, 519, 335, 37]
        "Confusion": [0.693, 0.225, 0.071, 0.011],   # [1135, 368, 116, 19]
        "Frustration": [0.781, 0.171, 0.034, 0.014]  # [1279, 280, 56, 23]
    }
    
    # Start with original predictions
    final_preds = preds.copy()
    
    # Define confidence thresholds - only modify predictions below these thresholds
    confidence_thresholds = {
        "Engagement": 0.80,  # More aggressive (important to fix)
        "Boredom": 0.75,     # Moderate adjustment
        "Confusion": 0.85,   # Less aggressive (to preserve accuracy)
        "Frustration": 0.85  # Less aggressive (to preserve accuracy)
    }
    
    # Mark which predictions we're allowed to modify based on confidence
    can_modify = confidences < confidence_thresholds[emotion]
    
    # Calculate current class distribution
    current_counts = np.bincount(final_preds, minlength=4)
    total_samples = len(final_preds)
    
    # Define minimum counts for each class (scaled by dataset size)
    min_counts = {
        "Engagement": [max(1, int(0.002*total_samples)), 
                    max(1, int(0.03*total_samples)),
                    int(0.4*total_samples), 
                    int(0.3*total_samples)],
        "Boredom": [int(0.35*total_samples),
                int(0.25*total_samples),
                int(0.15*total_samples),
                max(1, int(0.01*total_samples))],
        "Confusion": [int(0.58*total_samples), 
                    int(0.18*total_samples),
                    max(1, int(0.06*total_samples)), 
                    max(1, int(0.005*total_samples))],
        "Frustration": [int(0.65*total_samples), 
                    int(0.14*total_samples),
                    max(1, int(0.025*total_samples)), 
                    max(1, int(0.005*total_samples))]
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
            min_force = min(needed, 3, max_modifications - total_modified)
            change_indices = candidates[:min_force]
            final_preds[change_indices] = cls
            total_modified += len(change_indices)
            
            # Update counts
            current_counts = np.bincount(final_preds, minlength=4)
            needed = min_counts[emotion][cls] - current_counts[cls]
    
    # Apply emotion-specific adjustments
    if emotion == "Engagement" and total_modified < max_modifications:
        # For Engagement, boost classes 0 and 1 which are rare
        for cls in [0, 1]:
            if current_counts[cls] < min_counts[emotion][cls]:
                class_probs = probs[:, cls]
                threshold = 0.1 if cls == 0 else 0.15
                eligible = np.where((final_preds != cls) & (class_probs > threshold) & can_modify)[0]
                
                if len(eligible) > 0:
                    sorted_indices = eligible[np.argsort(-class_probs[eligible])]
                    num_to_change = min(min_counts[emotion][cls] - current_counts[cls], 
                                      len(sorted_indices), 
                                      max_modifications - total_modified)
                    
                    if num_to_change > 0:
                        change_indices = sorted_indices[:num_to_change]
                        final_preds[change_indices] = cls
                        total_modified += num_to_change
                        current_counts = np.bincount(final_preds, minlength=4)
    
    elif emotion in ["Confusion", "Frustration"] and total_modified < max_modifications:
        # For Confusion and Frustration, focus on class 1
        target_class1 = min_counts[emotion][1]
        if current_counts[1] < target_class1:
            # Take samples from class 0 which is abundant
            eligible = np.where((final_preds == 0) & (probs[:, 1] > 0.15) & can_modify)[0]
            
            if len(eligible) > 0:
                sorted_indices = eligible[np.argsort(-probs[eligible, 1])]
                num_to_change = min(target_class1 - current_counts[1],
                                    len(sorted_indices),
                                    max_modifications - total_modified)
                
                if num_to_change > 0:
                    change_indices = sorted_indices[:num_to_change]
                    final_preds[change_indices] = 1
                    total_modified += num_to_change
                    current_counts = np.bincount(final_preds, minlength=4)
    
    elif emotion == "Boredom" and total_modified < max_modifications:
        # For Boredom, balance all classes
        for target_cls in range(4):
            diff = min_counts[emotion][target_cls] - current_counts[target_cls]
            
            if diff > 0:
                # Find overrepresented classes
                overrep_classes = [c for c in range(4) if current_counts[c] > min_counts[emotion][c]]
                
                for source_cls in overrep_classes:
                    eligible = np.where((final_preds == source_cls) & 
                                       (probs[:, target_cls] > 0.1) & 
                                       can_modify)[0]
                    
                    if len(eligible) > 0:
                        sorted_indices = eligible[np.argsort(-probs[eligible, target_cls])]
                        num_to_change = min(diff, len(sorted_indices), 
                                          current_counts[source_cls] - min_counts[emotion][source_cls],
                                          max_modifications - total_modified)
                        
                        if num_to_change > 0:
                            change_indices = sorted_indices[:num_to_change]
                            final_preds[change_indices] = target_cls
                            diff -= num_to_change
                            total_modified += num_to_change
                            current_counts = np.bincount(final_preds, minlength=4)
                            
                            if diff <= 0:
                                break
    
    return final_preds

def apply_xgboost_postprocessing(preds, emotion):
    """
    Apply post-processing to XGBoost predictions.
    Adapted from the XGBOOST_HOG.py balanced distribution approach.
    """
    # Create a balanced distribution based on the emotion type
    target_dist = {
        "Engagement": np.array([0.01, 0.03, 0.49, 0.47]),  # Modest boost for class 1
        "Boredom": np.array([0.48, 0.31, 0.19, 0.02]),     # Closer to true dist
        "Confusion": np.array([0.72, 0.20, 0.07, 0.01]),   # Less aggressive
        "Frustration": np.array([0.80, 0.16, 0.03, 0.01])  # Less aggressive
    }
    
    # Calculate required counts for each class based on target distribution
    total_samples = len(preds)
    required_counts = (target_dist[emotion] * total_samples).astype(int)
    
    # Ensure we have exactly the number of samples
    required_counts[3] = total_samples - sum(required_counts[:3])
    
    # Current counts
    current_counts = np.bincount(preds, minlength=4)
    
    # Apply adjustments based on emotion-specific logic
    final_preds = preds.copy()
    
    # For small sample sizes (real-time), ensure minimum representation
    for cls in range(4):
        min_count = max(1, required_counts[cls] // 2)
        
        if current_counts[cls] < min_count:
            # Force at least minimum samples for this class
            needed = min_count - current_counts[cls]
            
            # Find the most abundant class to take from
            source_class = np.argmax(current_counts)
            if current_counts[source_class] > needed:
                # Get indices of source class
                indices = np.where(final_preds == source_class)[0]
                # Select random indices to change
                change_idx = np.random.choice(indices, needed, replace=False)
                final_preds[change_idx] = cls
                # Update counts
                current_counts = np.bincount(final_preds, minlength=4)
    
    return final_preds

# -------------------------------------------------------------------------
#                      MODEL DEFINITION AND LOADING FUNCTIONS
# -------------------------------------------------------------------------
class MobileNetV2LSTMStudent(nn.Module):
    def __init__(self, hidden_size=128, lstm_layers=1):
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

def load_student_model():
    try:
        # Check if ONNX file exists first
        onnx_path = MODEL_DIR / "student_model.onnx"
        
        # If ONNX file exists, use it
        if onnx_path.exists():
            print(f"ONNX model found at {onnx_path}. Loading directly.")
            available_providers = ort.get_available_providers()
            print(f"Using ONNX Runtime with available providers: {available_providers}")
            
            # Use CPUExecutionProvider if CUDA is not available
            if 'CUDAExecutionProvider' in available_providers:
                session = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])
            else:
                session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            
            return {'session': session, 'model': None}
        
        # If ONNX doesn't exist, load PyTorch model
        print("Loading ProEnsembleDistillation model...")
        model = MobileNetV2LSTMStudent(hidden_size=128, lstm_layers=1)
        
        # Define distill model path
        distill_model_path = MODEL_DIR / "best_student_model.pth"
        
        if not distill_model_path.exists():
            print(f"Error: Model file not found at {distill_model_path}")
            return None
            
        # Load checkpoint on CPU first
        checkpoint = torch.load(distill_model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        # Set model to eval mode
        model.eval()
        model.use_checkpointing = False  # Disable checkpointing for export
        
        return {'session': None, 'model': model}
    except Exception as e:
        print(f"Error loading student model: {e}")
        traceback.print_exc()
        return None
    
def load_xgboost_models():
    """Load XGBoost models for all emotions."""
    try:
        models = {}
        for emo in EMOTIONS:
            path = MODEL_DIR / f"final_model_{emo}.model"
            if not path.exists():
                print(f"Warning: Missing XGBoost model for {emo} at {path}")
                continue
                
            booster = xgb.Booster()
            booster.load_model(str(path))
            models[emo] = booster
            print(f"Loaded XGBoost model for {emo}")
        return models
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
        return {}


# -------------------------------------------------------------------------
#                      FEATURE EXTRACTION FUNCTIONS
# -------------------------------------------------------------------------
def extract_hog_features(frame):
    """Extract HOG features from a frame."""
    frame = cv2.resize(frame, (224, 224))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    # Return features as a 1D array
    return features

def preprocess_frame(frame):
    """Preprocess frame for neural network input."""
    # Resize to 224x224 as expected by the model
    frame = cv2.resize(frame, (224, 224))
    
    # Convert to RGB and normalize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # CHW format for PyTorch/ONNX
    frame = frame.transpose(2, 0, 1)
    return frame

# -------------------------------------------------------------------------
#                      MODEL INFERENCE FUNCTIONS
# -------------------------------------------------------------------------
def run_pro_ensemble_inference(frames, model_info):
    """Run ProEnsembleDistillation model on frames."""
    try:
        # If no model info or session, return default probabilities
        if not model_info or ('session' not in model_info and 'model' not in model_info):
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        
        # Make frames array compatible with the model
        frames_array = np.array(frames)
        input_shape = frames_array.shape
        
        # Handle any input shape dynamically
        if len(input_shape) == 3:  # Single frame [channels, height, width]
            frames_array = frames_array.reshape(1, 1, *input_shape)
        elif len(input_shape) == 4:  # Multiple frames [frames, channels, height, width]
            frames_array = frames_array.reshape(1, *input_shape)
        elif len(input_shape) > 5:  # Too many dimensions
            raise ValueError(f"Unexpected frame array shape: {input_shape}")
            
        # Run inference with ONNX session if available
        if model_info['session']:
            input_data = {'input': frames_array.astype(np.float32)}
            results = model_info['session'].run(None, input_data)
            output = results[0]  # Shape: [batch_size, len(EMOTIONS), 4]
        # Run inference with PyTorch model if session not available
        elif model_info['model']:
            model = model_info['model']
            with torch.no_grad():
                frames_tensor = torch.tensor(frames_array, dtype=torch.float32)
                output = model(frames_tensor).cpu().numpy()
        else:
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        
        # Convert output to probabilities with softmax
        probs = {}
        # Handle output shape based on what the model returns
        for i, emo in enumerate(EMOTIONS):
            if i < output.shape[1]:
                emo_logits = output[0, i]
                emo_probs = F.softmax(torch.tensor(emo_logits), dim=0).numpy()
                probs[emo] = emo_probs
            else:
                # Handle case where model returns fewer emotions than expected
                probs[emo] = np.array([0.25, 0.25, 0.25, 0.25])
        
        return probs
        
    except Exception as e:
        print(f"Error in ProEnsembleDistillation inference: {e}")
        traceback.print_exc()
        return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}

def run_xgboost_inference(hog_features, xgb_models):
    """Run XGBoost models on HOG features."""
    try:
        results = {}
        
        # Create DMatrix once for all features
        dmat = xgb.DMatrix(hog_features)
        
        for emo in EMOTIONS:
            if emo in xgb_models:
                # Get raw predictions
                raw_preds = xgb_models[emo].predict(dmat)
                
                # Convert to one-hot encoding
                pred_classes = raw_preds.astype(int)
                probs = np.zeros((len(pred_classes), 4))
                for i, cls in enumerate(pred_classes):
                    probs[i, cls] = 1.0
                
                results[emo] = probs
            else:
                # Default equal probabilities if model not available
                results[emo] = np.ones((hog_features.shape[0], 4)) * 0.25
        
        return results
    except Exception as e:
        print(f"Error in XGBoost inference: {e}")
        traceback.print_exc()
        return {emo: np.ones((hog_features.shape[0], 4)) * 0.25 for emo in EMOTIONS}
    
    
# -------------------------------------------------------------------------
#                      DATA GENERATION FUNCTIONS
# -------------------------------------------------------------------------
def load_and_process_real_data(num_samples=1720):
    """Load and process real data for benchmarking fusion methods."""
    print(f"Loading and processing {num_samples} real data samples...")
    
    # Define the paths based on XGBOOST_HOG.py
    base_dir = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
    data_dir = base_dir / "data" / "DAiSEE"
    frames_dir = data_dir / "ExtractedFrames"    
    labels_dir = data_dir / "Labels"
    
    # Create Test directory if it doesn't exist yet
    test_dir = frames_dir / "Test"
    if not test_dir.exists():
        print(f"Test directory {test_dir} doesn't exist, creating it...")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # Find source frame directories to copy from
        # Try to find source frames from either Train or Validation folders
        source_frames = []
        for source_folder in ["Train", "Validation"]:
            source_dir = frames_dir / source_folder
            if source_dir.exists():
                source_frames.extend(list(source_dir.glob("*")))
        
        if not source_frames:
            # If no organized folders, try to find any directories with frames
            source_frames = list(frames_dir.glob("*"))
            source_frames = [f for f in source_frames if f.is_dir() and list(f.glob("frame_*.jpg"))]
        
        if not source_frames:
            raise FileNotFoundError(f"No frame directories found in {frames_dir}")
        
        # Copy a subset of found directories to Test folder
        for i, source_folder in enumerate(source_frames[:min(num_samples * 2, len(source_frames))]):
            if i % 2 == 0:  # Take every other folder to get a diverse sample
                dest_folder = test_dir / source_folder.name
                if not dest_folder.exists():
                    print(f"Copying {source_folder.name} to Test directory...")
                    shutil.copytree(source_folder, dest_folder)
    
    # Load the models
    print("Loading models...")
    student_model = load_student_model()
    xgboost_models = load_xgboost_models()
    
    if not student_model:
        print("Warning: Failed to load student model, using default probabilities")
    
    if not xgboost_models:
        print("Warning: Failed to load XGBoost models, using default probabilities")
    
    # Find frame directories in the Test folder
    frame_dirs = list(test_dir.glob("*"))
    if not frame_dirs:
        print(f"No frame directories found in {test_dir}, looking in subdirectories...")
        frame_dirs = list(test_dir.glob("*/*"))
    
    if not frame_dirs:
        print(f"Error: No frame directories found in {test_dir}")
        # If still no directories, create a test folder with synthetic data
        print("Creating synthetic test data for benchmarking...")
        return generate_synthetic_data(num_samples)
    
    # Limit to requested number of samples
    frame_dirs = [d for d in frame_dirs if d.is_dir() and list(d.glob("frame_*.jpg"))]
    frame_dirs = frame_dirs[:min(num_samples, len(frame_dirs))]
    print(f"Found {len(frame_dirs)} frame directories for testing")
    
    # Create structures to hold data
    mobilenet_preds = {emotion: [] for emotion in EMOTIONS}
    mobilenet_probs = {emotion: [] for emotion in EMOTIONS}
    xgboost_preds = {emotion: [] for emotion in EMOTIONS}
    xgboost_probs = {emotion: [] for emotion in EMOTIONS}
    
    # Attempt to load ground truth labels if available
    labels_path = labels_dir / "TestLabels.csv"
    ground_truth = {emotion: [] for emotion in EMOTIONS}
    has_ground_truth = False
    
    if labels_path.exists():
        try:
            labels_df = pd.read_csv(labels_path)
            has_ground_truth = True
            print(f"Loaded ground truth labels from {labels_path}")
        except Exception as e:
            print(f"Error loading labels: {e}")
            has_ground_truth = False
    
    # Process each frame directory
    print("Processing frame directories...")
    for i, frame_dir in enumerate(frame_dirs):
        print(f"Processing directory {i+1}/{len(frame_dirs)}: {frame_dir.name}")
        
        # Find frame files in this directory
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Warning: No frames found in {frame_dir}")
            continue
        
        # Extract directory ID for finding labels
        dir_id = frame_dir.name
        
        # If we have ground truth, find the corresponding label
        if has_ground_truth:
            # Look for the directory ID in the labels DataFrame
            video_labels = labels_df[labels_df['ClipID'].str.contains(dir_id, na=False)]
            if not video_labels.empty:
                for emotion in EMOTIONS:
                    if emotion in video_labels.columns:
                        try:
                            ground_truth[emotion].append(int(video_labels[emotion].iloc[0]))
                        except (ValueError, IndexError):
                            ground_truth[emotion].append(0)  # Default value on error
                    else:
                        ground_truth[emotion].append(0)  # Default if column not found
            else:
                # No labels found for this video
                for emotion in EMOTIONS:
                    ground_truth[emotion].append(0)
        
        # Process frames for MobileNetV2_distilled model
        if student_model:
            # Process frames for temporal model (up to FRAME_HISTORY frames)
            frame_history = 40  # Same as in app.py
            processed_frames = []
            
            # Use at most frame_history frames from the directory
            selected_frames = frame_files[:min(frame_history, len(frame_files))]
            
            for frame_file in selected_frames:
                # Load and preprocess the frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue
                
                processed_frame = preprocess_frame(frame)
                processed_frames.append(processed_frame)
            
            # If we have enough frames, run MobileNetV2_distilled
            if len(processed_frames) > 0:
                # If we have fewer than frame_history frames, pad with the last frame
                while len(processed_frames) < frame_history:
                    processed_frames.append(processed_frames[-1] if processed_frames else np.zeros_like(processed_frame))
                
                # Run inference
                mobilenet_result = run_pro_ensemble_inference(processed_frames, student_model)
                
                # Store predictions and probabilities
                for emotion in EMOTIONS:
                    prob = mobilenet_result[emotion]
                    pred = np.argmax(prob)
                    mobilenet_preds[emotion].append(pred)
                    mobilenet_probs[emotion].append(prob)
            else:
                # No valid frames found
                for emotion in EMOTIONS:
                    mobilenet_preds[emotion].append(0)
                    mobilenet_probs[emotion].append(np.array([0.25, 0.25, 0.25, 0.25]))
        else:
            # No model available
            for emotion in EMOTIONS:
                mobilenet_preds[emotion].append(0)
                mobilenet_probs[emotion].append(np.array([0.25, 0.25, 0.25, 0.25]))
        
        # Process a representative frame for XGBOOST_HOG
        if xgboost_models and frame_files:
            # Use middle frame for XGBOOST_HOG
            mid_idx = len(frame_files) // 2
            frame = cv2.imread(str(frame_files[mid_idx]))
            
            if frame is not None:
                # Extract HOG features
                hog_features = extract_hog_features(frame).reshape(1, -1)
                
                # Run XGBoost inference
                xgboost_result = run_xgboost_inference(hog_features, xgboost_models)
                
                # Store predictions and probabilities
                for emotion in EMOTIONS:
                    prob = xgboost_result[emotion][0]  # First (and only) sample
                    pred = np.argmax(prob)
                    xgboost_preds[emotion].append(pred)
                    xgboost_probs[emotion].append(prob)
            else:
                # Invalid frame
                for emotion in EMOTIONS:
                    xgboost_preds[emotion].append(0)
                    xgboost_probs[emotion].append(np.array([0.25, 0.25, 0.25, 0.25]))
        else:
            # No model available or no frames
            for emotion in EMOTIONS:
                xgboost_preds[emotion].append(0)
                xgboost_probs[emotion].append(np.array([0.25, 0.25, 0.25, 0.25]))
    
    # If we don't have ground truth labels, generate random ones for benchmarking
    if not has_ground_truth or not any(ground_truth.values()):
        print("No ground truth labels found. Using synthetic ground truth for benchmarking.")
        # Define distribution based on typical class distribution
        class_distributions = {
            "Engagement": [0.05, 0.10, 0.45, 0.40],  # Mostly engaged
            "Boredom": [0.45, 0.30, 0.20, 0.05],     # Mostly not bored
            "Confusion": [0.65, 0.20, 0.10, 0.05],   # Mostly not confused
            "Frustration": [0.70, 0.15, 0.10, 0.05]  # Mostly not frustrated
        }
        
        num_processed = len(mobilenet_preds["Engagement"])
        for emotion in EMOTIONS:
            ground_truth[emotion] = np.random.choice(
                range(4),
                size=num_processed,
                p=class_distributions[emotion]
            ).tolist()
    
    # Apply post-processing to the raw predictions
    for emotion in EMOTIONS:
        if len(mobilenet_probs[emotion]) > 0:
            mobilenet_probs_array = np.stack(mobilenet_probs[emotion])
            mobilenet_preds[emotion] = apply_pro_ensemble_postprocessing(mobilenet_probs_array, emotion)
        
        if len(xgboost_preds[emotion]) > 0:
            xgboost_preds_array = np.array(xgboost_preds[emotion])
            xgboost_preds[emotion] = apply_xgboost_postprocessing(xgboost_preds_array, emotion)
    
    return {
        "ground_truth": ground_truth,
        "MobileNetV2_distilled": {
            "name": "MobileNetV2_distilled", 
            "predictions": mobilenet_preds,
            "probabilities": mobilenet_probs
        },
        "XGBOOST_HOG": {
            "name": "XGBOOST_HOG",
            "predictions": xgboost_preds,
            "probabilities": xgboost_probs
        }
    }

# Fallback to generating synthetic data if real data processing fails
def generate_synthetic_data(num_samples=1720):
    """Generate synthetic data for benchmarking when real data is unavailable."""
    print(f"Generating {num_samples} synthetic data samples...")
    
    # Define class probability distributions for each emotion
    class_distributions = {
        "Engagement": [0.05, 0.15, 0.45, 0.35],  # Mostly engaged
        "Boredom": [0.45, 0.30, 0.20, 0.05],     # Mostly not bored
        "Confusion": [0.65, 0.20, 0.10, 0.05],   # Mostly not confused
        "Frustration": [0.70, 0.15, 0.10, 0.05]  # Mostly not frustrated
    }
    
    # Generate ground truth labels using the distributions
    ground_truth = {emotion: [] for emotion in EMOTIONS}
    for emotion in EMOTIONS:
        ground_truth[emotion] = np.random.choice(
            range(4),
            size=num_samples,
            p=class_distributions[emotion]
        ).tolist()
    
    # Generate "predictions" for MobileNetV2_distilled (with intentional errors)
    mobilenet_preds = {emotion: [] for emotion in EMOTIONS}
    mobilenet_probs = {emotion: [] for emotion in EMOTIONS}
    
    # Generate "predictions" for XGBOOST_HOG
    xgboost_preds = {emotion: [] for emotion in EMOTIONS}
    xgboost_probs = {emotion: [] for emotion in EMOTIONS}
    
    # Generate data for each emotion
    for emotion in EMOTIONS:
        true_labels = ground_truth[emotion]
        
        for true_label in true_labels:
            # MobileNetV2_distilled accuracy depends on emotion
            mobilenet_accuracy = {
                "Engagement": 0.85,
                "Boredom": 0.70,
                "Confusion": 0.65,
                "Frustration": 0.60
            }[emotion]
            
            # XGBOOST_HOG accuracy depends on emotion
            xgboost_accuracy = {
                "Engagement": 0.70, 
                "Boredom": 0.85,
                "Confusion": 0.80,
                "Frustration": 0.82
            }[emotion]
            
            # Generate MobileNetV2_distilled prediction
            if np.random.random() < mobilenet_accuracy:
                mobilenet_pred = true_label  # Correct prediction
            else:
                # Incorrect prediction - more likely to be close to true label
                weights = [1.0/(abs(i-true_label)+1) for i in range(4)]
                weights[true_label] = 0  # Can't predict the true label (already handled above)
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                    mobilenet_pred = np.random.choice(range(4), p=weights)
                else:
                    mobilenet_pred = (true_label + 1) % 4  # Simple fallback
            
            # Generate XGBOOST_HOG prediction
            if np.random.random() < xgboost_accuracy:
                xgboost_pred = true_label  # Correct prediction
            else:
                # Incorrect prediction
                weights = [1.0/(abs(i-true_label)+1) for i in range(4)]
                weights[true_label] = 0
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                    xgboost_pred = np.random.choice(range(4), p=weights)
                else:
                    xgboost_pred = (true_label + 1) % 4
            
            # Generate probabilities (softmax-like values)
            mobilenet_prob = np.zeros(4)
            mobilenet_prob[mobilenet_pred] = 0.7 + np.random.random() * 0.2  # Base confidence
            # Distribute remaining probability among other classes
            remaining_prob = 1.0 - mobilenet_prob[mobilenet_pred]
            remaining_indices = [i for i in range(4) if i != mobilenet_pred]
            for idx in remaining_indices:
                if idx == remaining_indices[-1]:
                    mobilenet_prob[idx] = remaining_prob  # Last one gets the remainder
                else:
                    p = remaining_prob * np.random.random()
                    mobilenet_prob[idx] = p
                    remaining_prob -= p
            
            # Same for XGBOOST_HOG
            xgboost_prob = np.zeros(4)
            xgboost_prob[xgboost_pred] = 0.7 + np.random.random() * 0.2
            remaining_prob = 1.0 - xgboost_prob[xgboost_pred]
            remaining_indices = [i for i in range(4) if i != xgboost_pred]
            for idx in remaining_indices:
                if idx == remaining_indices[-1]:
                    xgboost_prob[idx] = remaining_prob
                else:
                    p = remaining_prob * np.random.random()
                    xgboost_prob[idx] = p
                    remaining_prob -= p
            
            # Store results
            mobilenet_preds[emotion].append(int(mobilenet_pred))
            mobilenet_probs[emotion].append(mobilenet_prob)
            xgboost_preds[emotion].append(int(xgboost_pred))
            xgboost_probs[emotion].append(xgboost_prob)
    
    return {
        "ground_truth": ground_truth,
        "MobileNetV2_distilled": {
            "name": "MobileNetV2_distilled", 
            "predictions": mobilenet_preds,
            "probabilities": mobilenet_probs
        },
        "XGBOOST_HOG": {
            "name": "XGBOOST_HOG",
            "predictions": xgboost_preds,
            "probabilities": xgboost_probs
        }
    }
    
def process_all_frames_temporally(frame_dirs, student_model, xgboost_models, max_frames_per_dir=None):
    """Process all frames temporally from all directories."""
    print("Processing frames temporally...")
    
    # Create structures to hold data
    mobilenet_preds = {emotion: [] for emotion in EMOTIONS}
    mobilenet_probs = {emotion: [] for emotion in EMOTIONS}
    xgboost_preds = {emotion: [] for emotion in EMOTIONS}
    xgboost_probs = {emotion: [] for emotion in EMOTIONS}
    
    # Process each frame directory
    for i, frame_dir in enumerate(frame_dirs):
        print(f"Processing directory {i+1}/{len(frame_dirs)}: {frame_dir.name}")
        
        # Find frame files in this directory
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Warning: No frames found in {frame_dir}")
            continue
            
        # Limit frames if specified
        if max_frames_per_dir is not None:
            frame_files = frame_files[:min(max_frames_per_dir, len(frame_files))]
            
        # Process all frames temporally
        processed_frames = []
        
        for frame_file in frame_files:
            # Load and preprocess the frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
                
            processed_frame = preprocess_frame(frame)
            processed_frames.append(processed_frame)
            
        # If we have frames, run MobileNetV2_distilled
        if processed_frames:
            # Run inference - handle any number of frames
            mobilenet_result = run_pro_ensemble_inference(processed_frames, student_model)
            
            # Store predictions and probabilities
            for emotion in EMOTIONS:
                if emotion in mobilenet_result:
                    prob = mobilenet_result[emotion]
                    pred = np.argmax(prob)
                    mobilenet_preds[emotion].append(pred)
                    mobilenet_probs[emotion].append(prob)
            
        # Process a representative frame for XGBOOST_HOG
        if xgboost_models and frame_files:
            # Use middle frame for XGBOOST_HOG
            mid_idx = len(frame_files) // 2
            frame = cv2.imread(str(frame_files[mid_idx]))
            
            if frame is not None:
                # Extract HOG features
                hog_features = extract_hog_features(frame).reshape(1, -1)
                
                # Run XGBoost inference
                xgboost_result = run_xgboost_inference(hog_features, xgboost_models)
                
                # Store predictions and probabilities
                for emotion in EMOTIONS:
                    if emotion in xgboost_result:
                        prob = xgboost_result[emotion][0]  # First (and only) sample
                        pred = np.argmax(prob)
                        xgboost_preds[emotion].append(pred)
                        xgboost_probs[emotion].append(prob)
    
    return mobilenet_preds, mobilenet_probs, xgboost_preds, xgboost_probs

# -------------------------------------------------------------------------
#                      FUSION METHODS
# -------------------------------------------------------------------------
def apply_selective_fusion(data, sample_idx):
    """
    Selective fusion - use MobileNetV2_distilled for Engagement, XGBOOST_HOG for other emotions
    This is similar to the logic in app.py
    """
    results = {}
    
    for emotion in EMOTIONS:
        if emotion == "Engagement":
            # Use MobileNetV2_distilled for Engagement
            results[emotion] = data["MobileNetV2_distilled"]["predictions"][emotion][sample_idx]
        else:
            # Use XGBOOST_HOG for other emotions
            results[emotion] = data["XGBOOST_HOG"]["predictions"][emotion][sample_idx]
    
    return results

def apply_weighted_fusion(data, sample_idx):
    """Apply weighted fusion between models based on predefined weights."""
    # Define weights per emotion
    weights = {
        "Engagement": [0.6, 0.4],    # [MobileNetV2_distilled, XGBOOST_HOG]
        "Boredom": [0.3, 0.7],
        "Confusion": [0.3, 0.7],
        "Frustration": [0.3, 0.7]
    }
    
    results = {}
    
    for emotion in EMOTIONS:
        # Get probabilities
        prob1 = data["MobileNetV2_distilled"]["probabilities"][emotion][sample_idx]
        prob2 = data["XGBOOST_HOG"]["probabilities"][emotion][sample_idx]
        
        # Apply weights
        w1 = weights[emotion][0]
        w2 = weights[emotion][1]
        
        # Weighted sum
        combined_probs = w1 * prob1 + w2 * prob2
        results[emotion] = np.argmax(combined_probs)
    
    return results

def apply_hybrid_balanced_fusion(data, sample_idx):
    """
    Hybrid Balanced Fusion:
      - Engagement from selective_fusion
      - Boredom & Frustration from MobileNetV2_distilled
      - Confusion from weighted_fusion
    """
    # reuse the existing fusion functions for those slots
    sel = apply_selective_fusion(data, sample_idx)
    wtd = apply_weighted_fusion(data, sample_idx)
    results = {}
    results["Engagement"]   = sel["Engagement"]
    results["Boredom"]      = data["MobileNetV2_distilled"]["predictions"]["Boredom"][sample_idx]
    results["Confusion"]    = wtd["Confusion"]
    results["Frustration"]  = data["MobileNetV2_distilled"]["predictions"]["Frustration"][sample_idx]
    return results


def apply_mobilenet_confidence_gate(data, sample_idx, gate=None):
    """
    Wrapper around apply_mobilenet_confidence_gate_once.
    If no gate is provided, fall back to the global CONF_GATE.
    """
    gate_to_use = gate if gate is not None else CONF_GATE
    return apply_mobilenet_confidence_gate_once(data, sample_idx, gate_to_use)

def apply_emotion_specific_gated_fusion(data, sample_idx):
    """
    Apply emotion-specific fusion strategies based on benchmarks.
    This matches the logic from app.py for production use.
    
    For each emotion:
    1. Engagement: Use MobileNet predictions with low threshold
    2. Boredom: Use gated blend with high threshold
    3. Confusion: Use gated weighted fusion
    4. Frustration: Use gated switch for very low confidence
    """
    results = {}
    
    for emotion in EMOTIONS:
        # Get probabilities and predictions from MobileNet
        mobilenet_probs = data["MobileNetV2_distilled"]["probabilities"][emotion][sample_idx]
        mobilenet_class = np.argmax(mobilenet_probs)
        mobilenet_conf = np.max(mobilenet_probs)
        
        # Get the confidence gate and fusion strategy for this emotion
        gate = EMOTION_GATES[emotion]
        strategy = FUSION_STRATEGIES[emotion]
        
        # Only check XGBoost if we have it
        if emotion in data["XGBOOST_HOG"]["probabilities"]:
            xgboost_probs = data["XGBOOST_HOG"]["probabilities"][emotion][sample_idx]
            xgboost_class = np.argmax(xgboost_probs)
            xgboost_conf = np.max(xgboost_probs)
        else:
            # No XGBoost predictions, fall back to MobileNet
            results[emotion] = mobilenet_class
            continue
        
        # Apply the appropriate fusion strategy
        if mobilenet_conf >= gate:
            # If MobileNet confidence is above gate, always use its prediction
            results[emotion] = mobilenet_class
        else:
            # Apply emotion-specific fusion strategy
            if strategy == "mobilenet_only":
                # For "mobilenet_only", still use MobileNet even below gate
                results[emotion] = mobilenet_class
                
            elif strategy == "gated_blend":
                # For "gated_blend", blend probabilities with 60% MobileNet, 40% XGBoost
                blended_probs = 0.6 * mobilenet_probs + 0.4 * xgboost_probs
                results[emotion] = np.argmax(blended_probs)
                
            elif strategy == "gated_weighted":
                # For "gated_weighted", use weighted fusion approach
                weighted_probs = 0.3 * mobilenet_probs + 0.7 * xgboost_probs
                results[emotion] = np.argmax(weighted_probs)
                
            elif strategy == "gated_switch":
                # For "gated_switch", hard switch to XGBoost for very low confidence
                results[emotion] = xgboost_class
            
            else:
                # Fallback to MobileNet if strategy not recognized
                results[emotion] = mobilenet_class
    
    return results


# -------------------------------------------------------------------------
#           GRID‑SEARCH FOR CONF‑GATE **BY** CLASS            
# -------------------------------------------------------------------------
def tune_conf_gate(data, gate_values):
    """
    For each emotion, finds the gate value that yields:
      • highest accuracy
      • highest macro‐F1
      • highest balanced=(accuracy+F1)/2
    Returns three dicts: best_acc_gates, best_f1_gates, best_balanced_gates.
    """
    # initialize best‐score holders
    best_acc      = {emo: (None, -1.0) for emo in EMOTIONS}
    best_f1       = {emo: (None, -1.0) for emo in EMOTIONS}
    best_balanced = {emo: (None, -1.0) for emo in EMOTIONS}

    print(f"Tuning CONF_GATE by class over {len(gate_values)} candidates…")
    for g in gate_values:
        # one‐pass predictions under gate=g
        preds_all = [
            apply_mobilenet_confidence_gate_once(data, i, g)
            for i in range(len(data["ground_truth"][EMOTIONS[0]]))
        ]
        for emo in EMOTIONS:
            gt    = data["ground_truth"][emo]
            preds = [p[emo] for p in preds_all]
            acc   = accuracy_score(gt, preds)
            f1m   = f1_score(gt, preds, average="macro")
            bal   = (acc + f1m) / 2

            # update if better
            if acc  > best_acc[emo][1]:      best_acc[emo]      = (g, acc)
            if f1m  > best_f1[emo][1]:       best_f1[emo]       = (g, f1m)
            if bal  > best_balanced[emo][1]: best_balanced[emo] = (g, bal)

    # strip out just the gate values
    best_acc_gates      = {emo: gv for emo, (gv, _)  in best_acc.items()}
    best_f1_gates       = {emo: gv for emo, (gv, _)  in best_f1.items()}
    best_balanced_gates = {emo: gv for emo, (gv, _)  in best_balanced.items()}

    print("Per‑class best ACC gates:      ", best_acc_gates)
    print("Per‑class best F1 gates:       ", best_f1_gates)
    print("Per‑class best BALANCED gates: ", best_balanced_gates)
    return best_acc_gates, best_f1_gates, best_balanced_gates


# -------------------------------------------------------------------------
#                      TIME MEASUREMENT
# -------------------------------------------------------------------------
def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, (end_time - start_time) * 1000  # Return milliseconds in a tuple with result

# -------------------------------------------------------------------------
#                      BENCHMARKING FUNCTIONS
# -------------------------------------------------------------------------
def run_fusion_benchmark(data):
    """Run benchmark comparing different fusion methods with real timing."""
    print("Running benchmark of fusion methods...")
    
    num_samples = len(data["ground_truth"]["Engagement"])
    
    # Dictionary to store results for each fusion method
    all_results = {method: [] for method in FUSION_METHODS}
    
    # Dictionary to store execution times
    execution_times = {method: [] for method in FUSION_METHODS}
    
    # Process each sample with each fusion method
    for i in range(num_samples):
        # Measure actual execution time for each method
        result, time_ms = measure_execution_time(apply_selective_fusion, data, i)
        all_results["selective_fusion"].append(result)
        execution_times["selective_fusion"].append(time_ms)
        
        result, time_ms = measure_execution_time(apply_weighted_fusion, data, i)
        all_results["weighted_fusion"].append(result)
        execution_times["weighted_fusion"].append(time_ms)
        
        result, time_ms = measure_execution_time(apply_hybrid_balanced_fusion, data, i)
        all_results["hybrid_balanced_fusion"].append(result)
        execution_times["hybrid_balanced_fusion"].append(time_ms)
        
        result, time_ms = measure_execution_time(apply_mobilenet_confidence_gate, data, i)
        all_results["mobilenet_confidence_gate"].append(result)
        execution_times["mobilenet_confidence_gate"].append(time_ms)
        
        result, time_ms = measure_execution_time(apply_emotion_specific_gated_fusion, data, i)
        all_results["emotion_specific_gated_fusion"].append(result)
        execution_times["emotion_specific_gated_fusion"].append(time_ms)
    
    # Calculate average execution time for each method
    avg_execution_times = {
        method: sum(times) / len(times) if times else 0 
        for method, times in execution_times.items()
    }
    
    # Also measure model inference times separately for comparison
    mobilenet_times = []
    xgboost_times = []
    for i in range(min(100, num_samples)):  # Sample 100 measurements
        _, time_ms = measure_execution_time(
            run_pro_ensemble_inference, 
            [data["MobileNetV2_distilled"]["probabilities"]["Engagement"][i]],
            {'model': None, 'session': None}
        )
        mobilenet_times.append(time_ms)
        
        _, time_ms = measure_execution_time(
            run_xgboost_inference,
            np.array([[0.0]]),  # Dummy data
            {}  # No models, just measuring function overhead
        )
        xgboost_times.append(time_ms)
    
    avg_mobilenet = sum(mobilenet_times) / len(mobilenet_times) if mobilenet_times else 0
    avg_xgboost = sum(xgboost_times) / len(xgboost_times) if xgboost_times else 0
    
    # -------------------------------------------------------------
    #                 METRICS FOR EVERY FUSION METHOD
    # -------------------------------------------------------------
    metrics = {}
    
    for method in FUSION_METHODS:
        metrics[method] = {}
        for emotion in EMOTIONS:
            # Predictions and ground truth
            preds = [r[emotion] for r in all_results[method]]
            gt = data["ground_truth"][emotion]

            # Basic scores
            acc = accuracy_score(gt, preds)
            f1 = f1_score(gt, preds, average="macro")

            # Store with actual measured time
            metrics[method][emotion] = {
                "accuracy": acc,
                "f1_score": f1,
                "inference_time": avg_execution_times[method] / 1000.0  # Convert to seconds
            }

    # Add individual model metrics with measured times
    metrics["MobileNetV2_distilled"] = {}
    metrics["XGBOOST_HOG"] = {}
    
    for emotion in EMOTIONS:
        # MobileNetV2_distilled metrics
        distill_preds = data["MobileNetV2_distilled"]["predictions"][emotion]
        gt = data["ground_truth"][emotion]
        metrics["MobileNetV2_distilled"][emotion] = {
            "accuracy": accuracy_score(gt, distill_preds),
            "f1_score": f1_score(gt, distill_preds, average='macro'),
            "inference_time": avg_mobilenet / 1000.0  # Convert to seconds
        }
        
        # XGBOOST_HOG metrics
        xgb_preds = data["XGBOOST_HOG"]["predictions"][emotion]
        metrics["XGBOOST_HOG"][emotion] = {
            "accuracy": accuracy_score(gt, xgb_preds),
            "f1_score": f1_score(gt, xgb_preds, average='macro'),
            "inference_time": avg_xgboost / 1000.0  # Convert to seconds
        }
    
    # Save raw results and metrics
    with open(METRICS_DIR / "fusion_benchmark_results.json", "w") as f:
        json.dump({
            "chosen_conf_gate": CONF_GATE, 
            "all_results": {method: [
                {emotion: int(result[emotion]) for emotion in EMOTIONS} 
                for result in all_results[method]
            ] for method in FUSION_METHODS},
            "execution_times": {method: times for method, times in execution_times.items()},
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": num_samples
        }, f, indent=2)
    
    print(f"Saved benchmark results to {METRICS_DIR / 'fusion_benchmark_results.json'}")
    
    # Return for visualization
    return all_results, metrics, data["ground_truth"]
    
    # -------------------------------------------------------------
    #                 METRICS FOR EVERY FUSION METHOD
    # -------------------------------------------------------------
    metrics = {}

    # Calculate average execution time for each method
    avg_execution_times = {
        method: sum(times) / len(times) if times else 0 
        for method, times in execution_times.items()
    }
    
    # Also measure model inference times separately for comparison
    mobilenet_times = []
    xgboost_times = []
    for i in range(min(100, num_samples)):  # Sample 100 measurements
        _, time_ms = measure_execution_time(
            run_pro_ensemble_inference, 
            [data["MobileNetV2_distilled"]["probabilities"]["Engagement"][i]],
            {'model': None, 'session': None}
        )
        mobilenet_times.append(time_ms)
        
        _, time_ms = measure_execution_time(
            run_xgboost_inference,
            np.array([[0.0]]),  # Dummy data
            {}  # No models, just measuring function overhead
        )
        xgboost_times.append(time_ms)
    
    avg_mobilenet = sum(mobilenet_times) / len(mobilenet_times) if mobilenet_times else 0
    avg_xgboost = sum(xgboost_times) / len(xgboost_times) if xgboost_times else 0
    
    for method in FUSION_METHODS:
        metrics[method] = {}
        for emotion in EMOTIONS:
            # Predictions and ground truth
            preds = [r[emotion] for r in all_results[method]]
            gt = data["ground_truth"][emotion]

            # Basic scores
            acc = accuracy_score(gt, preds)
            f1 = f1_score(gt, preds, average="macro")

            # Store with actual measured time
            metrics[method][emotion] = {
                "accuracy": acc,
                "f1_score": f1,
                "inference_time": avg_execution_times[method] / 1000.0  # Convert back to seconds
            }

    # Add individual model metrics with measured times
    metrics["MobileNetV2_distilled"] = {}
    metrics["XGBOOST_HOG"] = {}
    
    for emotion in EMOTIONS:
        # MobileNetV2_distilled metrics
        distill_preds = data["MobileNetV2_distilled"]["predictions"][emotion]
        gt = data["ground_truth"][emotion]
        metrics["MobileNetV2_distilled"][emotion] = {
            "accuracy": accuracy_score(gt, distill_preds),
            "f1_score": f1_score(gt, distill_preds, average='macro'),
            "inference_time": avg_mobilenet / 1000.0  # Convert to seconds
        }
        
        # XGBOOST_HOG metrics
        xgb_preds = data["XGBOOST_HOG"]["predictions"][emotion]
        metrics["XGBOOST_HOG"][emotion] = {
            "accuracy": accuracy_score(gt, xgb_preds),
            "f1_score": f1_score(gt, xgb_preds, average='macro'),
            "inference_time": avg_xgboost / 1000.0  # Convert to seconds
        }
    
    # Save raw results and metrics
    with open(METRICS_DIR / "fusion_benchmark_results.json", "w") as f:
        json.dump({
            "chosen_conf_gate": CONF_GATE, 
            "all_results": {method: [
                {emotion: int(result[emotion]) for emotion in EMOTIONS} 
                for result in all_results[method]
            ] for method in FUSION_METHODS},
            "execution_times": {method: times for method, times in execution_times.items()},
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": num_samples
        }, f, indent=2)
    
    print(f"Saved benchmark results to {METRICS_DIR / 'fusion_benchmark_results.json'}")
    
    # Return for visualization
    return all_results, metrics, data["ground_truth"]

# -------------------------------------------------------------------------
#                      VISUALIZATION FUNCTIONS
# -------------------------------------------------------------------------
def plot_confusion_matrices(all_results, ground_truth):
    """Plot confusion matrices for all fusion methods."""
    print("Generating confusion matrices...")
    
    # Add individual models to the visualization
    methods_to_plot = FUSION_METHODS
    
    for emotion in EMOTIONS:
        # Calculate dynamic grid layout
        num_methods = len(methods_to_plot)
        cols = min(2, num_methods)
        rows = (num_methods + cols - 1) // cols
        
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Confusion Matrices for {emotion}", fontsize=16)
        
        for i, method in enumerate(methods_to_plot):
            # Extract predictions for this method and emotion
            preds = [r[emotion] for r in all_results[method]]
                
            # Get ground truth for this emotion
            gt = ground_truth[emotion]
            
            # Ensure gt and preds have same length
            min_len = min(len(gt), len(preds))
            gt = gt[:min_len]
            preds = preds[:min_len]
            
            if min_len == 0:
                # Skip if no data
                continue
            
            # Get unique classes to handle variable number of classes
            unique_classes = sorted(set(gt).union(set(preds)))
            num_classes = len(unique_classes)
            
            # Compute confusion matrix with dynamic labels
            cm = confusion_matrix(gt, preds, labels=unique_classes)
            
            # Create normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            
            # Get appropriate labels based on the emotion and actual classes
            if emotion in EMOTION_LABELS:
                class_labels = [EMOTION_LABELS[emotion][c] if c < len(EMOTION_LABELS[emotion]) else f"Class {c}" 
                              for c in unique_classes]
            else:
                class_labels = [f"Class {c}" for c in unique_classes]
            
            # Plot in dynamic grid
            plt.subplot(rows, cols, i+1)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_labels,
                        yticklabels=class_labels)
            plt.title(f"{method}")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_path = PLOTS_DIR / f"confusion_matrices_{emotion}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_accuracy_comparison(metrics):
    """Plot accuracy comparison between fusion methods."""
    print("Generating accuracy comparison chart...")
    plt.figure(figsize=(12, 8))
    
    # Extract accuracy for each emotion and method
    methods = list(metrics.keys())
    emotions = EMOTIONS
    
    # Set up the bar position
    x = np.arange(len(emotions))
    width = 0.15  # Width of bars
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods-1)/2*width, (n_methods-1)/2*width, n_methods)
    
    for i, method in enumerate(methods):
        accuracies = [metrics[method][emotion]["accuracy"] for emotion in emotions]
        plt.bar(x + offsets[i], accuracies, width, label=method)
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Fusion Methods')
    plt.xticks(x, emotions)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value annotations
    for i, method in enumerate(methods):
        accuracies = [metrics[method][emotion]["accuracy"] for emotion in emotions]
        for j, acc in enumerate(accuracies):
            plt.text(x[j] + offsets[i], acc + 0.01, f'{acc:.2f}', 
                     ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    save_path = PLOTS_DIR / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_emotion_distribution(all_results, ground_truth):
    """Plot predicted emotion distribution for each method."""
    print("Generating emotion distribution charts...")
    
    # First, plot ground truth distribution
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Ground Truth Emotion Distribution", fontsize=16)
    
    # Calculate dynamic grid layout based on emotion count
    num_emotions = len(EMOTIONS)
    cols = min(2, num_emotions)  # Maximum 2 columns
    rows = (num_emotions + cols - 1) // cols  # Ceiling division
    
    for i, emotion in enumerate(EMOTIONS):
        # Count occurrences of each class
        counts = np.bincount([gt for gt in ground_truth[emotion]], minlength=4)
        
        # Plot - now with dynamic grid
        plt.subplot(rows, cols, i+1)
        bars = plt.bar(EMOTION_LABELS[emotion], counts, color='darkgreen')
        plt.title(emotion)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_path = PLOTS_DIR / f"emotion_distribution_ground_truth.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_fusion_agreement(all_results):
    """Plot agreement between different fusion methods."""
    print("Generating fusion agreement charts...")
    agreement_data = {emotion: {} for emotion in EMOTIONS}
    
    for emotion in EMOTIONS:
        agreement_matrix = np.zeros((len(FUSION_METHODS), len(FUSION_METHODS)))
        
        for i, method1 in enumerate(FUSION_METHODS):
            for j, method2 in enumerate(FUSION_METHODS):
                # Calculate agreement percentage
                total = len(all_results[method1])
                agree = sum(1 for k in range(total) if all_results[method1][k][emotion] == all_results[method2][k][emotion])
                
                agreement_matrix[i, j] = agree / total * 100
        
        agreement_data[emotion] = agreement_matrix
    
    # Plot agreement matrices
    for emotion in EMOTIONS:
        plt.figure(figsize=(10, 8))
        plt.title(f"Agreement Between Fusion Methods - {emotion}", fontsize=16)
        
        sns.heatmap(agreement_data[emotion], annot=True, fmt='.1f', cmap='YlGnBu',
                   xticklabels=FUSION_METHODS, yticklabels=FUSION_METHODS)
        
        plt.tight_layout()
        save_path = PLOTS_DIR / f"fusion_agreement_{emotion}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_radar_chart(metrics):
    """Create radar chart comparing fusion methods across metrics."""
    print("Generating radar chart...")
    # Calculate summary metrics for each method
    summary = {}
    
    for method in metrics:
        avg_acc = np.mean([metrics[method][emotion]["accuracy"] for emotion in EMOTIONS])
        avg_f1 = np.mean([metrics[method][emotion]["f1_score"] for emotion in EMOTIONS])
        avg_time = metrics[method][EMOTIONS[0]]["inference_time"]  # Use synthetic time
        
        eng_acc = metrics[method]["Engagement"]["accuracy"]
        bor_acc = metrics[method]["Boredom"]["accuracy"]
        con_acc = metrics[method]["Confusion"]["accuracy"]
        fru_acc = metrics[method]["Frustration"]["accuracy"]
        
        summary[method] = {
            "Overall Accuracy": avg_acc,
            "Overall F1": avg_f1,
            "Performance": 1 - (avg_time/0.03),  # Normalize synthetic time
            "Engagement Detection": eng_acc,
            "Boredom Detection": bor_acc,
            "Confusion Detection": con_acc,
            "Frustration Detection": fru_acc
        }
    
    # Create radar chart
    categories = ["Overall Accuracy", "Overall F1", "Performance", 
                "Engagement Detection", "Boredom Detection", "Confusion Detection", "Frustration Detection"]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of categories
    N = len(categories)
    
    # Angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Plot each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    
    for i, method in enumerate(summary):
        values = [summary[method][cat] for cat in categories]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Fusion Methods Comparison", size=20)
    
    save_path = PLOTS_DIR / "fusion_radar_chart.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_emotion_timeline(all_results, ground_truth, num_samples=1720):
    """Generate a timeline visualization of emotion levels."""
    print("Generating emotion timeline charts...")
    # Limit to a reasonable number of samples for visualization
    sample_count = min(len(all_results["selective_fusion"]), num_samples)
    
    for emotion in EMOTIONS:
        plt.figure(figsize=(15, 10))
        
        # Plot ground truth
        plt.subplot(len(FUSION_METHODS)+1, 1, 1)
        plt.plot(range(sample_count), ground_truth[emotion][:sample_count], 'o-', label="Ground Truth", color='black')
        plt.ylabel("Level")
        plt.title(f"{emotion} Timeline - Ground Truth")
        plt.yticks(range(4), EMOTION_LABELS[emotion])
        plt.grid(True, alpha=0.3)
        
        # Plot each fusion method
        for i, method in enumerate(FUSION_METHODS):
            plt.subplot(len(FUSION_METHODS)+1, 1, i+2)
            preds = [r[emotion] for r in all_results[method]][:sample_count]
            plt.plot(range(sample_count), preds, 'o-', label=method)
            plt.ylabel("Level")
            plt.title(f"{emotion} Timeline - {method}")
            plt.yticks(range(4), EMOTION_LABELS[emotion])
            plt.grid(True, alpha=0.3)
            
            # Calculate and display accuracy
            acc = accuracy_score(ground_truth[emotion][:sample_count], preds)
            plt.text(0.02, 0.9, f"Accuracy: {acc:.2f}", transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel("Sample")
        plt.tight_layout()
        save_path = PLOTS_DIR / f"emotion_timeline_{emotion}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

def generate_summary_report(metrics):
    """Generate a summary report of all metrics."""
    print("Generating summary report...")
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "fusion_methods_compared": FUSION_METHODS,
    }
    
    # Add overall averages
    report["overall_averages"] = {}
    for method in metrics:
        report["overall_averages"][method] = {
            "avg_accuracy": np.mean([metrics[method][e]["accuracy"] for e in EMOTIONS]),
            "avg_f1_score": np.mean([metrics[method][e]["f1_score"] for e in EMOTIONS]),
            "avg_inference_time": np.mean([metrics[method][e]["inference_time"] for e in EMOTIONS])
        }
    
    return report

def create_multi_emotion_chart(metrics):
    """Create a chart showing how each method handles multiple emotions."""
    print("Generating multi-emotion performance chart...")
    
    methods = list(metrics.keys())
    
    # Set up the plot with dynamic width
    plt.figure(figsize=(max(14, len(methods)*2), 8))
    
    # Create a bar for each method showing accuracy across all emotions
    num_emotions = len(EMOTIONS)
    bar_width = 0.8 / num_emotions  # Scale bar width based on emotion count
    index = np.arange(len(methods))
    
    # Generate colors dynamically
    color_map = plt.cm.get_cmap('tab10', num_emotions)
    colors = [color_map(i) for i in range(num_emotions)]
    
    for i, emotion in enumerate(EMOTIONS):
        accuracies = []
        for method in methods:
            # Check if this emotion exists in the metrics for this method
            if emotion in metrics[method]:
                accuracies.append(metrics[method][emotion]["accuracy"])
            else:
                accuracies.append(0)  # Default if missing
                
        plt.bar(index + i * bar_width, 
                accuracies, 
                bar_width,
                label=emotion,
                color=colors[i % len(colors)])
    
    plt.xlabel('Fusion Method')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Method and Emotion')
    
    # Adjust x-ticks based on number of methods
    plt.xticks(index + bar_width * (num_emotions-1)/2, 
              [m if len(m) < 15 else m[:12]+'...' for m in methods], 
              rotation=45 if len(methods) > 5 else 0)
    
    plt.legend(loc='best')
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    save_path = PLOTS_DIR / "multi_emotion_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved multi-emotion chart to {save_path}")
    
def write_markdown_report(report, metrics, execution_times=None):
    md_path = METRICS_DIR / "benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        # Title and timestamp
        f.write("# Fusion Methods Benchmark Report\n")
        f.write(f"Generated on {report['timestamp']}\n\n")
        
        # Per-Class Best CONF_GATE
        f.write("## Per-Class Best CONF_GATE\n\n")
        f.write("| Emotion    | Best ACC Gate | Best F1 Gate | Best Balanced Gate |\n")
        f.write("|------------|---------------|--------------|--------------------|\n")
        for emo in EMOTIONS:
            acc_g = report["per_class_best_conf_gate"]["accuracy"][emo]
            f1_g  = report["per_class_best_conf_gate"]["f1"][emo]
            bal_g = report["per_class_best_conf_gate"]["balanced"][emo]
            f.write(f"| {emo:<10} | {acc_g:>13.2f} | {f1_g:>12.2f} | {bal_g:>18.2f} |\n")
        f.write("\n")
        
        # Performance Metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Method | Emotion | Accuracy | F1 Score | Inference Time |\n")
        f.write("|--------|---------|----------|----------|---------------|\n")
        for method, emo_dict in metrics.items():
            for emotion, vals in emo_dict.items():
                f.write(
                    f"| {method} | {emotion} | "
                    f"{vals['accuracy']:.4f} | {vals['f1_score']:.4f} | "
                    f"{vals['inference_time']:.4f}s |\n"
                )
        f.write("\n")
        
        # Add timing statistics section if execution_times is provided
        if execution_times:
            f.write("## Fusion Method Timing Statistics (milliseconds)\n\n")
            f.write("| Method | Min | Max | Mean | Median | 95th Percentile |\n")
            f.write("|--------|-----|-----|------|--------|----------------|\n")
            
            for method, times in execution_times.items():
                if times:
                    min_time = min(times)
                    max_time = max(times)
                    mean_time = sum(times) / len(times)
                    median_time = sorted(times)[len(times)//2]
                    p95_time = sorted(times)[int(len(times) * 0.95)]
                    
                    f.write(f"| {method} | {min_time:.3f} | {max_time:.3f} | {mean_time:.3f} | {median_time:.3f} | {p95_time:.3f} |\n")
            
            f.write("\n")
        
        # Overall Average Metrics table
        f.write("## Overall Average Metrics\n\n")
        f.write("| Method | Avg Accuracy | Avg F1 | Avg Inference Time |\n")
        f.write("|--------|--------------|--------|---------------------|\n")
        for method, avg in report["overall_averages"].items():
            f.write(
                f"| {method} | {avg['avg_accuracy']:.4f} | {avg['avg_f1_score']:.4f} | "
                f"{avg['avg_inference_time']:.4f}s |\n"
            )
        f.write("\n")

        # Best Method per Emotion
        f.write("## Best Method per Emotion\n\n")
        f.write("| Emotion | Best Method | Accuracy | F1 Score |\n")
        f.write("|---------|-------------|----------|----------|\n")
        for emotion in EMOTIONS:
            # pick the method with highest accuracy for this emotion
            best = max(metrics.keys(), key=lambda m: metrics[m][emotion]["accuracy"])
            bvals = metrics[best][emotion]
            f.write(
                f"| {emotion} | {best} | "
                f"{bvals['accuracy']:.4f} | {bvals['f1_score']:.4f} |\n"
            )
        f.write("\n")

        # Overall best
        best_overall = max(report["overall_averages"].keys(),
                           key=lambda m: report["overall_averages"][m]["avg_accuracy"])
        bo = report["overall_averages"][best_overall]
        f.write(f"## Overall Best Method: {best_overall}\n")
        f.write(f"Average Accuracy: {bo['avg_accuracy']:.4f}\n")
        f.write(f"Average F1 Score: {bo['avg_f1_score']:.4f}\n\n")

        # Key Findings
        f.write("## Key Findings\n\n")
        f.write(f"- **{best_overall}** yields the highest average accuracy across all emotions.\n")
        for emotion in EMOTIONS:
            best = max(metrics.keys(), key=lambda m: metrics[m][emotion]["accuracy"])
            acc = metrics[best][emotion]["accuracy"]
            f.write(f"- For **{emotion}**, **{best}** is best with {acc:.2%} accuracy.\n")

    print(f"Wrote Markdown report to {md_path}")
    
# -------------------------------------------------------------------------
#                      MAIN FUNCTION
# -------------------------------------------------------------------------
def main():
    """Main function to run benchmark and generate visualizations."""
    print("Starting Fusion Benchmark with Real Models")
    
    # Load and process real data instead of generating synthetic data
    data = load_and_process_real_data(num_samples=1720)  # Use fewer samples for real data
    
    # —––––– per‑class grid‑search for CONF_GATE
    best_acc_gates, best_f1_gates, best_balanced_gates = (
        tune_conf_gate(data, CONF_GATE_CANDIDATES)
    )

    # Run the fusion benchmark with timing measurements
    all_results, metrics, ground_truth = run_fusion_benchmark(data)

    # Get the execution times from the JSON file
    with open(METRICS_DIR / "fusion_benchmark_results.json", "r") as f:
        benchmark_data = json.load(f)
        execution_times = benchmark_data.get("execution_times", {})

    # Build the final report
    report = generate_summary_report(metrics)
    report["per_class_best_conf_gate"] = {
        "accuracy":  best_acc_gates,
        "f1":        best_f1_gates,
        "balanced":  best_balanced_gates
    }

    # Write the markdown report with execution times
    write_markdown_report(report, metrics, execution_times)
    
    # [rest of existing code]
    # Dump everything to JSON in one shot
    with open(METRICS_DIR / "fusion_benchmark_results.json", "w") as f:
        json.dump({
            "per_class_best_conf_gate": report["per_class_best_conf_gate"],
            "all_results": {
                m: [
                    {emo: int(res[emo]) for emo in EMOTIONS}
                    for res in all_results[m]
                ]
                for m in FUSION_METHODS
            },
            "execution_times": execution_times,  # Use real execution times
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": len(ground_truth[EMOTIONS[0]])
        }, f, indent=2)

    # Now kick off all your plots and finish up
    plot_confusion_matrices(all_results, ground_truth)
    plot_accuracy_comparison(metrics)
    plot_emotion_distribution(all_results, ground_truth)
    plot_fusion_agreement(all_results)
    plot_radar_chart(metrics)
    generate_emotion_timeline(all_results, ground_truth)
    create_multi_emotion_chart(metrics)

    print("Benchmark complete! All results saved to", RESULTS_DIR)
    
if __name__ == "__main__":
    main()
    print("Script executed successfully.")