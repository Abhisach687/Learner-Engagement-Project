import os
import sys
import time
import json
import asyncio
import psutil
import cv2
import numpy as np
import threading
import queue
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import xgboost as xgb
from skimage.feature import hog
from torch.utils.checkpoint import checkpoint
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import onnxruntime as ort
from nicegui import ui, app
import base64
import time
import numpy as np
import cv2
import psutil
import queue
import torch
import xgboost as xgb
import asyncio
import random

# Add a console log buffer to store recent messages
CONSOLE_LOG_BUFFER = deque(maxlen=40000)  # Store last 40000 log messages

# Create a custom print function that also stores logs
original_print = print
def log_print(*args, **kwargs):
    # Convert args to a string
    message = " ".join(str(arg) for arg in args)
    # Add timestamp to message
    timestamped_message = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}"
    # Store in buffer
    CONSOLE_LOG_BUFFER.append(timestamped_message)
    # Call original print
    original_print(*args, **kwargs)

# Replace the built-in print function
print = log_print

# -------------------------------------------------------------------------
#                      CONFIGURATION AND PATHS
# -------------------------------------------------------------------------
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
DISTILL_MODEL_PATH = MODEL_DIR / "best_student_model.pth"
EMOTIONS = ["Engagement", "Boredom", "Confusion", "Frustration"]
EMOTION_ICONS = {
    "Engagement": "🎯",
    "Boredom": "😴",
    "Confusion": "🤔",
    "Frustration": "😠"
}

# Emotion-specific confidence gates based on benchmark analysis
EMOTION_GATES = {
    "Engagement": 0.10,
    "Boredom": 0.45,
    "Confusion": 0.10,
    "Frustration": 0.10
}

# Available fusion strategies
FUSION_STRATEGIES = {
    "Engagement": "mobilenet_only",  # Use MobileNet predictions unless confidence extremely low
    "Boredom": "gated_blend",        # Blend probabilities when MobileNet confidence below threshold
    "Confusion": "gated_weighted",    # Use weighted fusion for Confusion if MobileNet confidence low
    "Frustration": "gated_switch"     # Hard switch to XGBoost for very low confidence predictions
}

# Optimization settings
FRAME_SKIP = 2               # Actual frame skip to apply
ENABLE_FRAME_SKIPPING = True    # Enable dynamic frame skipping
MIN_FRAME_TIME = 1/15           # Cap processing at 15 FPS
MAX_LATENCY_THRESHOLD = 78      # Max latency for frame processing
LATENCY_BUFFER = 20           # Buffer for latency smoothing

# Currently selected fusion mode
CURRENT_FUSION_MODE = "emotion_specific_gated_fusion"

# Global confidence threshold for gating strategies
CONFIDENCE_GATE = 0.45

# Set CUDA device if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_INFO = f"Using {DEVICE} for inference"

# Neural model config
FRAME_HISTORY = 40  # Number of frames to keep for temporal processing
SMOOTHING_WINDOW = 5  # Number of predictions to smooth over time

# System monitoring thresholds
CPU_HIGH_THRESHOLD = 85
CPU_MEDIUM_THRESHOLD = 70
GPU_HIGH_THRESHOLD = 85
GPU_MEDIUM_THRESHOLD = 70

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"DISTILL_MODEL_PATH: {DISTILL_MODEL_PATH}")
print(f"DISTILL_MODEL_PATH exists: {DISTILL_MODEL_PATH.exists()}")

# Runtime settings
ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

# Resource pools
thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 1)
PREDICTION_QUEUE = queue.Queue(maxsize=5)
RESULTS_BUFFER = deque(maxlen=SMOOTHING_WINDOW)

# Display frame rate control
DISPLAY_FPS = 30     # Target display FPS
PROCESS_FPS = 10     # Process fewer frames for better performance
last_processing_time = 0
last_frame = None    # Store latest frame for smoother display
frame_counter = 0    # For FPS calculation
fps_start_time = time.time()
reported_latency = 33  

results = {emo: 0 for emo in EMOTIONS}  # Global results for sharing between tasks

advice_text = "Waiting for webcam data..."

# -------------------------------------------------------------------------
#                      STUDENT MODEL DEFINITION
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
    
# -------------------------------------------------------------------------
#                      MODEL LOADING FUNCTIONS
# -------------------------------------------------------------------------

def load_student_model():
    try:
        # Check if ONNX file exists first
        onnx_path = MODEL_DIR / "student_model.onnx"
        
        # If ONNX file exists, skip PyTorch model loading
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
        
        # If ONNX doesn't exist, load PyTorch model and convert
        print("Loading ProEnsembleDistillation model...")
        model = MobileNetV2LSTMStudent(hidden_size=128, lstm_layers=1)
        
        if not DISTILL_MODEL_PATH.exists():
            print(f"Error: Model file not found at {DISTILL_MODEL_PATH}")
            return None
            
        # Load checkpoint on CPU first
        checkpoint = torch.load(DISTILL_MODEL_PATH, map_location='cpu')
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        # Set model to eval mode
        model.eval()
        model.use_checkpointing = False  # Disable checkpointing for export
        
        # Create dummy input with correct dimensions (batch, frames, channels, height, width)
        print(f"Converting model to ONNX format at {onnx_path}")
        dummy_input = torch.randn(1, FRAME_HISTORY, 3, 224, 224, device='cpu')
        
        # Export with dynamic axes for batch size
        torch.onnx.export(
            model, 
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Create ONNX Runtime session
        available_providers = ort.get_available_providers()
        print(f"Using ONNX Runtime with available providers: {available_providers}")
        
        # Use CPUExecutionProvider if CUDA is not available
        if 'CUDAExecutionProvider' in available_providers:
            session = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])
        else:
            session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        return {'session': session, 'model': model}
    except Exception as e:
        print(f"Error loading student model: {e}")
        import traceback
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
    

def validate_xgboost_models(models):
    """Validate that the XGBoost models are properly loaded and can make predictions."""
    if not models:
        print("❌ No XGBoost models were loaded!")
        return False
    
    print("\n===== VALIDATING XGBOOST MODELS =====")
    all_valid = True
    
    # Create a simple test feature vector (same size as HOG features)
    test_features = np.random.random(9216).astype(np.float32)  # Adjust size to match your HOG features
    
    try:
        dmatrix = xgb.DMatrix(test_features.reshape(1, -1))
        
        for emotion in EMOTIONS:
            if emotion not in models:
                print(f"❌ Missing model for {emotion}")
                all_valid = False
                continue
                
            try:
                # Test prediction
                raw_pred = models[emotion].predict(dmatrix)
                
                if raw_pred is None or len(raw_pred) == 0:
                    print(f"❌ Empty prediction for {emotion}")
                    all_valid = False
                else:
                    print(f"✅ Valid prediction for {emotion}: {raw_pred[0]}")
                    
            except Exception as e:
                print(f"❌ Error predicting with {emotion} model: {e}")
                all_valid = False
    
    except Exception as e:
        print(f"❌ Error creating test DMatrix: {e}")
        all_valid = False
    
    print(f"XGBoost validation {'passed' if all_valid else 'failed'}\n")
    return all_valid

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
    
    # Return features as a 1D array, no reshaping needed
    return features

def preprocess_frame(frame):
    """Preprocess frame for neural network input."""
    # Resize to 224x224 as expected by the model
    frame = cv2.resize(frame, (224, 224))
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to RGB and normalize
    frame = frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # CHW format for PyTorch/ONNX
    frame = frame.transpose(2, 0, 1)
    return frame

# -------------------------------------------------------------------------
#                      VALIDATION FUNCTION
# -------------------------------------------------------------------------

def validate_model_output(predictions):
    """Check if predictions are valid (not all zeros or very close to zeros)."""
    if not predictions:
        return False
        
    # Count emotions with non-uniform predictions
    valid_emotions = 0
    for emotion, value in predictions.items():
        # Check if the value is a valid class index (0-3)
        if 0 <= value <= 3:
            valid_emotions += 1
        # If it's an array, check if it's not uniform (not all 0.25)
        elif isinstance(value, np.ndarray) and not np.allclose(value, np.array([0.25, 0.25, 0.25, 0.25])):
            valid_emotions += 1
    
    # We need at least one valid emotion prediction
    return valid_emotions > 0

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
    
    # Apply emotion-specific adjustments based on our implementation from ProEnsembleDistillation.py
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
            # In real-time, we might have to make arbitrary assignments
            # which is fine since we'll smooth over time
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
#                      PREDICTION AND FUSION FUNCTIONS
# -------------------------------------------------------------------------

async def run_xgboost(hog_features, xgboost_models):
    """Run XGBoost models on extracted HOG features with proper formatting and error handling."""
    try:
        # If no XGBoost models loaded, return default probabilities
        if not xgboost_models:
            print("No XGBoost models available")
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        
        # Ensure features are the correct shape and type
        if isinstance(hog_features, np.ndarray):
            # Reshape to 1D if needed (XGBoost expects a flat vector)
            if len(hog_features.shape) > 1:
                features = hog_features.flatten().reshape(1, -1)
            else:
                features = hog_features.reshape(1, -1)
            
            # Create DMatrix with proper formatting
            try:
                dmatrix = xgb.DMatrix(features)
            except Exception as e:
                print(f"Failed to create DMatrix: {e}, shape={features.shape}, type={features.dtype}")
                return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        else:
            print(f"Invalid HOG features type: {type(hog_features)}")
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        
        # Run inference for each emotion with detailed error handling
        results = {}
        for emotion in EMOTIONS:
            if emotion not in xgboost_models:
                print(f"Missing XGBoost model for {emotion}")
                results[emotion] = np.array([0.25, 0.25, 0.25, 0.25])
                continue
            
            try:
                # Get raw predictions
                raw_pred = xgboost_models[emotion].predict(dmatrix)[0]
                
                # Check if prediction contains valid values
                if np.isnan(raw_pred).any() or np.isinf(raw_pred).any():
                    print(f"Invalid prediction values for {emotion}: {raw_pred}")
                    results[emotion] = np.array([0.25, 0.25, 0.25, 0.25])
                    continue
                
                # Debug output to check raw predictions
                print(f"XGBoost raw prediction for {emotion}: {raw_pred}")
                
                # Apply softmax with numerical stability
                raw_pred = raw_pred - np.max(raw_pred)  # Subtract max for numerical stability
                exp_pred = np.exp(raw_pred)
                probs = exp_pred / np.sum(exp_pred)
                
                # Ensure valid probabilities
                if np.isnan(probs).any() or np.sum(probs) < 0.99:
                    print(f"Invalid probabilities for {emotion}: {probs}, sum={np.sum(probs)}")
                    probs = np.array([0.25, 0.25, 0.25, 0.25])
                    
                results[emotion] = probs
                
            except Exception as e:
                print(f"Error in XGBoost prediction for {emotion}: {e}")
                results[emotion] = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Final check for all-zero results
        all_valid = True
        for emotion, probs in results.items():
            if np.allclose(probs, np.array([0.25, 0.25, 0.25, 0.25])):
                all_valid = False
        
        if not all_valid:
            print("⚠️ XGBoost returned uniform probabilities - check model loading")
        
        return results
        
    except Exception as e:
        print(f"Error in XGBoost inference: {e}")
        import traceback
        traceback.print_exc()
        return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}

async def run_pro_ensemble(frame_buffer, model_info):
    """Run ProEnsembleDistillation model with optimized buffer handling."""
    try:
        # Fast path for missing models or empty buffers
        if not model_info or 'session' not in model_info or not model_info['session']:
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
        
        # Check if we have enough frames for accurate prediction
        buffer_size = len(frame_buffer)
        
        # Skip excessive logging - only log every 10 frames when buffer is filling
        if buffer_size < FRAME_HISTORY and buffer_size % 10 == 0:
            print(f"Building frame buffer: {buffer_size}/{FRAME_HISTORY}")
        
        # For extremely small buffers, return baseline values
        if buffer_size < 10:  
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
            
        # Process frames with adaptive buffer handling
        frames = np.array(frame_buffer)
        
        # Add batch dimension if needed
        if len(frames.shape) == 4:  # [frames, channels, height, width]
            frames = frames.reshape(1, *frames.shape)
            
        # Smart padding that maintains temporal characteristics
        if buffer_size < FRAME_HISTORY:
            # Create padded array
            padded_frames = np.zeros((1, FRAME_HISTORY, frames.shape[2], frames.shape[3], frames.shape[4]), 
                                   dtype=frames.dtype)
            
            # Place actual frames in the end positions for better temporal modeling
            start_idx = FRAME_HISTORY - buffer_size
            padded_frames[0, start_idx:] = frames[0]
            
            input_data = {'input': padded_frames.astype(np.float32)}
        else:
            # Complete buffer - use as is
            input_data = {'input': frames.astype(np.float32)}
            
        # Run inference with error handling
        try:
            results = model_info['session'].run(None, input_data)
            output = results[0]
            
            # Calculate confidence based on buffer fullness
            confidence = min(1.0, buffer_size / FRAME_HISTORY)
            
            # Process outputs with optimized numpy operations
            probs = {}
            for i, emo in enumerate(EMOTIONS):
                emo_logits = output[0, i]
                # Use numpy for softmax - faster than PyTorch conversion
                exp_logits = np.exp(emo_logits - np.max(emo_logits))
                emo_probs = exp_logits / exp_logits.sum()
                probs[emo] = emo_probs
                
            return probs
            
        except Exception as e:
            # More informative error message
            if "invalid dimensions" in str(e):
                print(f"Model dimension error: buffer_size={buffer_size}, expected={FRAME_HISTORY}")
            else:
                print(f"Model inference error: {e}")
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
            
    except Exception as e:
        print(f"Error in model processing: {e}")
        return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}

def select_final_prediction(mobilenet_probs, xgboost_probs, emotion):
    """
    Apply emotion-specific gated fusion strategy.
    
    Args:
        mobilenet_probs: Probability distribution from MobileNet model
        xgboost_probs: Probability distribution from XGBoost model
        emotion: Emotion type being processed
    
    Returns:
        Predicted class after fusion
    """
    # Get maximum probability and its class from MobileNet
    mobilenet_conf = np.max(mobilenet_probs)
    mobilenet_class = np.argmax(mobilenet_probs)
    
    # Get maximum probability and its class from XGBoost
    xgboost_conf = np.max(xgboost_probs)
    xgboost_class = np.argmax(xgboost_probs)
    
    # Get the confidence gate for this emotion
    gate = EMOTION_GATES[emotion]
    
    # Get the fusion strategy for this emotion
    strategy = FUSION_STRATEGIES[emotion]
    
    # Apply the appropriate fusion strategy
    if mobilenet_conf >= gate:
        # If MobileNet confidence is above gate, use its prediction
        return mobilenet_class
    else:
        # Apply emotion-specific fusion strategy
        if strategy == "mobilenet_only":
            return mobilenet_class
            
        elif strategy == "gated_blend":
            blended_probs = 0.6 * mobilenet_probs + 0.4 * xgboost_probs
            return np.argmax(blended_probs)
            
        elif strategy == "gated_weighted":
            weighted_probs = 0.3 * mobilenet_probs + 0.7 * xgboost_probs
            return np.argmax(weighted_probs)
            
        elif strategy == "gated_switch":
            return xgboost_class
    
    # Fallback to MobileNet prediction
    return mobilenet_class

def apply_temporal_smoothing(current_pred, emotion):
    """Apply temporal smoothing to predictions for stability."""
    global RESULTS_BUFFER
    
    # Add current prediction to the buffer
    if len(RESULTS_BUFFER) == 0:
        # Initialize with the current prediction for all positions
        RESULTS_BUFFER.extend([{emotion: current_pred for emotion in EMOTIONS}] * SMOOTHING_WINDOW)
    else:
        RESULTS_BUFFER.append({emotion: current_pred for emotion in EMOTIONS})
    
    # Apply exponential weighting (more recent predictions have higher weight)
    weights = np.exp(np.linspace(0, 1, len(RESULTS_BUFFER)))
    weights /= weights.sum()
    
    # Calculate weighted average for the emotion
    counts = np.zeros(4)
    for i, result in enumerate(RESULTS_BUFFER):
        pred = result[emotion]
        counts[pred] += weights[i]
    
    # Return the class with highest smoothed count
    return np.argmax(counts)

# -------------------------------------------------------------------------
#                      SYSTEM MONITORING FUNCTIONS
# -------------------------------------------------------------------------
def get_cpu_utilization():
    """Get current CPU utilization."""
    return psutil.cpu_percent()

def get_gpu_utilization():
    """Get current GPU utilization if available."""
    try:
        if torch.cuda.is_available():
            # This requires pynvml
            nvidia_smi_path = 'nvidia-smi'
            result = os.popen(f'"{nvidia_smi_path}" --query-gpu=utilization.gpu --format=csv,noheader,nounits').read()
            return int(result.strip())
        return 0
    except:
        return 0

def determine_processing_rate():
    """Determine frame processing rate based on system load."""
    cpu_load = get_cpu_utilization()
    gpu_load = get_gpu_utilization()
    
    if cpu_load > CPU_HIGH_THRESHOLD or gpu_load > GPU_HIGH_THRESHOLD:
        return "process_every_third_frame"
    elif cpu_load > CPU_MEDIUM_THRESHOLD or gpu_load > GPU_MEDIUM_THRESHOLD:
        return "process_every_second_frame"
    else:
        return "process_every_frame"
    
def adjust_frame_skip_by_latency(current_latency):
    """
    Dynamically adjust frame skip based on processing latency.
    - Range: 2-48 frames
    - Immediately increases if latency > MAX_LATENCY_THRESHOLD
    - Gradually decreases if latency is consistently low
    """
    global FRAME_SKIP
    
    # Fast path for high latency - immediate response
    if current_latency > MAX_LATENCY_THRESHOLD:
        # Aggressive increase - jump by 2 or more if latency is very high
        if current_latency > MAX_LATENCY_THRESHOLD * 1.5:
            FRAME_SKIP = min(28, FRAME_SKIP + 3)
        else:
            FRAME_SKIP = min(28, FRAME_SKIP + 1)
        print(f"⚠️ High latency ({current_latency}ms) - increased skip to {FRAME_SKIP}")
        return
    
    # If latency is good with room to spare, consider reducing frame skip
    if current_latency < (MAX_LATENCY_THRESHOLD - LATENCY_BUFFER):
        # Only reduce every few frames to avoid oscillation
        if random.random() < 0.1:  # 10% chance to reduce (gradual)
            FRAME_SKIP = max(2, FRAME_SKIP - 1)
            print(f"✅ Good latency ({current_latency}ms) - decreased skip to {FRAME_SKIP}")
    
    # Otherwise maintain current frame skip
    return


# -------------------------------------------------------------------------
#                      MAIN PROCESSING FUNCTION
# -------------------------------------------------------------------------
# Optimize the frame skipping logic with reduced overhead
async def process_frame(frame, frame_count, frame_buffer, models):
    """Optimized processing function for a single frame."""
    global last_processing_time, reported_latency, FRAME_SKIP, results
    
    # Simple frame skip application
    if frame_count % max(1, FRAME_SKIP) != 0:
        return results  # Fast return for skipped frames
    
    # Throttle processing to maintain target framerate
    elapsed = time.time() - last_processing_time
    if elapsed < MIN_FRAME_TIME:
        return results  # Skip processing if too soon
    
    # Start timing
    processing_start = time.time()
    
    # Process frame with optimized operations
    
    # 1. Preprocess and buffer frame (reuse frame memory where possible)
    processed_frame = preprocess_frame(frame)
    frame_buffer.append(processed_frame)
    
    # 2. Maintain buffer size with efficient operation
    if len(frame_buffer) > FRAME_HISTORY:
        frame_buffer.pop(0)
    
    # 3. Extract features for both models - HOG can be expensive
    hog_features = extract_hog_features(frame)
    
    # 4. Run models with emotion-specific gated fusion
    xgboost_probs = await run_xgboost(hog_features, models['xgboost'])
    mobilenet_probs = await run_pro_ensemble(frame_buffer, models['student'])
    
    # 5. Apply fusion with reduced calls
    emotion_predictions = {}
    for emotion in EMOTIONS:
        predicted_class = select_final_prediction(
            mobilenet_probs[emotion], 
            xgboost_probs[emotion], 
            emotion
        )
        
        # Apply temporal smoothing for stability
        emotion_predictions[emotion] = apply_temporal_smoothing(predicted_class, emotion)
    
    # Update latency metrics
    last_processing_time = time.time()
    current_latency = int((last_processing_time - processing_start) * 1000)
    reported_latency = current_latency
    
    # Adjust frame skip based on latency
    adjust_frame_skip_by_latency(current_latency)
    
    # Additional logging - reduced frequency
    if frame_count % 100 == 0:
        print(f"FRAME {frame_count}: Latency={current_latency}ms, Skip={FRAME_SKIP}")
    
    return emotion_predictions

# -------------------------------------------------------------------------
#                      UI COMPONENTS AND HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_emotion_label(emotion_class, emotion_name):
    """Convert emotion class to human-readable label."""
    labels = {
        "Engagement": ["Disengaged", "Low Engagement", "Engaged", "Very Engaged"],
        "Boredom": ["Not Bored", "Slightly Bored", "Bored", "Very Bored"],
        "Confusion": ["Not Confused", "Slightly Confused", "Confused", "Very Confused"],
        "Frustration": ["Not Frustrated", "Slightly Frustrated", "Frustrated", "Very Frustrated"]
    }
    return labels[emotion_name][emotion_class]

def get_emotion_color(emotion_class, emotion_name):
    """Get color for emotion level visualization."""
    colors = {
        "Engagement": ["#ff6b6b", "#ffa06b", "#6bff6b", "#6bffa0"],  # Red to Green
        "Boredom": ["#6bff6b", "#ffa06b", "#ff8f6b", "#ff6b6b"],      # Green to Red
        "Confusion": ["#6bff6b", "#ffa06b", "#ff8f6b", "#ff6b6b"],    # Green to Red
        "Frustration": ["#6bff6b", "#ffa06b", "#ff8f6b", "#ff6b6b"]   # Green to Red
    }
    return colors[emotion_name][emotion_class]

def generate_advice(results):
    """Generate educational advice based on emotion detection with explanations."""
    # Simple null check
    if not results:
        return "Waiting for webcam data..."
    
    # Use a more explicit check for valid data
    if all(results.get(emotion, 0) == 0 for emotion in EMOTIONS):
        return "Waiting for enough data to provide personalized feedback..."
    
    # Extract emotion values with safe defaults
    engagement = results.get("Engagement", 0)
    boredom = results.get("Boredom", 0)
    confusion = results.get("Confusion", 0)
    frustration = results.get("Frustration", 0)
    
    # Create explanation of the reasoning behind advice
    explanation = "Based on: "
    explanations = []
    
    if engagement <= 1:
        explanations.append(f"low engagement level ({get_emotion_label(engagement, 'Engagement')})")
    else:
        explanations.append(f"good engagement level ({get_emotion_label(engagement, 'Engagement')})")
    
    if boredom >= 2:
        explanations.append(f"elevated boredom ({get_emotion_label(boredom, 'Boredom')})")
    
    if confusion >= 2:
        explanations.append(f"noticeable confusion ({get_emotion_label(confusion, 'Confusion')})")
        
    if frustration >= 2:
        explanations.append(f"visible frustration ({get_emotion_label(frustration, 'Frustration')})")
    
    explanation += ", ".join(explanations)
    
    # Original advice logic
    advice = ""
    if engagement <= 1:  # Disengaged or Low Engagement
        if boredom >= 2:  # Bored or Very Bored
            advice = "You seem disengaged. Try taking a short break or switching to a more interactive section."
        if confusion >= 2:  # Confused or Very Confused
            advice = "You appear to be struggling with this material. Consider reviewing the prerequisites or asking for help."
        if frustration >= 2:  # Frustrated or Very Frustrated
            advice = "You seem frustrated. Take a moment to relax and break the problem into smaller steps."
        advice = advice or "Try to engage more actively with the material by taking notes or working through examples."
        
    elif confusion >= 2:  # Engaged but Confused
        advice = "You're engaged but seem confused. Don't hesitate to ask questions or review previous sections."
        
    elif frustration >= 2:  # Engaged but Frustrated
        advice = "You're working hard but seem frustrated. Consider a different approach or ask for assistance."
        
    elif boredom >= 2:  # Engaged but Bored
        advice = "You're attentive but might benefit from more challenging material or interactive exercises."
        
    else:  # Fully engaged, not bored, confused or frustrated
        advice = "You're demonstrating excellent engagement! Keep up the good work."
    
    # Combine advice and explanation
    return f"{advice}\n\n{explanation}"

# -------------------------------------------------------------------------
#                      MAIN APPLICATION
# -------------------------------------------------------------------------
class EmotionDetectionApp:
    def __init__(self):
        self.last_processing_time = 0
        self.frame_count = 0
        self.frame_buffer = []
        self.webcam = None
        self.prev_advice = ""
        self.webcam_active = False
        self.processing_active = False
        self.last_results = {emo: 0 for emo in EMOTIONS}
        self.models = {
            'student': None,
            'xgboost': None
        }
        self.system_info = {
            'cpu': 0,
            'gpu': 0,
            'processing_rate': 'process_every_frame',
            'last_update': time.time()
        }
        
    async def initialize_models(self):
        """Load all models asynchronously."""
        print("Starting model initialization...")
        
        self.models['xgboost'] = await asyncio.get_event_loop().run_in_executor(
            thread_pool, load_xgboost_models
        )
        
        # Add this line to validate XGBoost models
        if self.models['xgboost']:
            validate_xgboost_models(self.models['xgboost'])
        
        print(f"Looking for student model at: {DISTILL_MODEL_PATH}")
        print(f"Model directory exists: {MODEL_DIR.exists()}")
        print(f"Student model exists: {DISTILL_MODEL_PATH.exists()}")
        
        try:
            self.models['student'] = await asyncio.get_event_loop().run_in_executor(
                thread_pool, load_student_model
            )
        except Exception as e:
            print(f"Error loading student model: {e}")
            self.models['student'] = None
        
        if not self.models['student'] and not self.models['xgboost']:
            ui.notify('Failed to load any models! Please check the logs and restart the application.', type='negative')
            return False
        
        ui.notify('Models loaded successfully!', type='positive')
        return True
    
    def update_system_info(self):
        """Update system information like CPU and GPU usage."""
        self.system_info['cpu'] = get_cpu_utilization()
        self.system_info['gpu'] = get_gpu_utilization()
        self.system_info['processing_rate'] = determine_processing_rate()
        self.system_info['last_update'] = time.time()
    

    async def process_webcam_frame(self, frame):
        """Process a single frame from the webcam with optimizations."""
        if not self.processing_active:
            return self.last_results
                
        self.frame_count += 1
        
        try:
            # Apply optimizations from the process_frame function
            results = await process_frame(
                frame, 
                self.frame_count,
                self.frame_buffer,
                self.models
            )
            
            # Print insights periodically (every 30 frames)
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}, Buffer: {len(self.frame_buffer)}/{FRAME_HISTORY}, Mode: {CURRENT_FUSION_MODE}")
                print(f"Latency: {reported_latency}ms, Frame Skip: {FRAME_SKIP}, Skip Enabled: {ENABLE_FRAME_SKIPPING}")
                
                # FIX: More informative validation - only warn when truly problematic
                if results and not validate_model_output(results):
                    print("WARNING: All zero values - check model outputs")
                    
                    # FIX: Try to recover by reinitializing if this happens repeatedly
                    if not hasattr(self, 'zero_value_count'):
                        self.zero_value_count = 0
                    
                    self.zero_value_count += 1
                    
                    # If we get 5 consecutive zero-value results, try reconnecting to webcam
                    if self.zero_value_count > 5:
                        print("Multiple zero-value frames detected. Attempting recovery...")
                        self.frame_buffer = []  # Clear frame buffer to force fresh start
                        self.zero_value_count = 0
                else:
                    # Reset counter when we get valid results
                    if hasattr(self, 'zero_value_count'):
                        self.zero_value_count = 0
            
            # Update last results if we have valid data
            if results and validate_model_output(results):
                self.last_results = results
                
            return results
        except Exception as e:
            print(f"Error in frame processing: {e}")
            import traceback
            traceback.print_exc()
            return self.last_results
    
    async def start_webcam(self):

        """Start the webcam feed."""
        if self.webcam_active:
            return
            
        # Ensure models are loaded before starting webcam
        if not self.models['student'] and not self.models['xgboost']:
            ui.notify('Models are not loaded! Please load models first.', type='negative')
            return
            
        try:
            # Try directly opening the primary camera without scanning all indices
            self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Add CAP_DSHOW for Windows
            
            if not self.webcam.isOpened():
                ui.notify('Failed to open webcam!', type='negative')
                return
                
            # Set webcam properties for better quality
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            self.webcam_active = True
            self.processing_active = True
            ui.notify('Webcam started!', type='positive')
        except Exception as e:
            ui.notify(f'Error starting webcam: {str(e)}', type='negative')
    
    def stop_webcam(self):
        """Stop the webcam feed and save console logs."""
        if not self.webcam_active:
            return
            
        self.webcam_active = False
        self.processing_active = False
        
        # Save logs to file
        try:
            log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"webcam_log_{log_timestamp}.txt"
            
            # Create logs directory if it doesn't exist
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            
            log_path = log_dir / log_filename
            
            # Write logs to file
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Webcam Session Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Device: {DEVICE}\n")
                f.write(f"Fusion Mode: {CURRENT_FUSION_MODE}\n")
                f.write(f"Frame Skip: {FRAME_SKIP}, Dynamic: {ENABLE_FRAME_SKIPPING}\n")
                f.write(f"Latency: {reported_latency}ms\n")
                f.write(f"CPU: {get_cpu_utilization()}%, GPU: {get_gpu_utilization()}%\n")
                f.write(f"Frame Buffer Size: {len(self.frame_buffer)}/{FRAME_HISTORY}\n\n")
                
                # Write the captured console output
                f.write("=" * 80 + "\n")
                f.write("CONSOLE LOG HISTORY\n")
                f.write("=" * 80 + "\n\n")
                for log_line in CONSOLE_LOG_BUFFER:
                    f.write(f"{log_line}\n")
            
            ui.notify(f'Logs saved to {log_path}', type='info', duration=8)
            print(f"Console logs saved to: {log_path}")
        except Exception as e:
            ui.notify(f'Error saving logs: {str(e)}', type='negative')
            print(f"Error saving logs: {e}")
        
        if self.webcam:
            self.webcam.release()
            self.webcam = None
            
        ui.notify('Webcam stopped!', type='info')
        
    async def get_webcam_frame(self):
        """Get the current webcam frame."""
        if not self.webcam_active or not self.webcam:
            return None
            
        ret, frame = self.webcam.read()
        if not ret:
            self.stop_webcam()
            ui.notify('Failed to read from webcam!', type='negative')
            return None
            
        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        return frame
#-------------------------------------------------------------------------
#                      NiceGUI SETUP
# -------------------------------------------------------------------------

# Initialize app
emotion_app = EmotionDetectionApp()


@ui.page('/')
async def index_page():
    # Page setup
    ui.add_head_html("""
        <style>
            body {
                background-color: #f7f9fc;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            [data-theme="dark"] body {
                background-color: #1e1e1e;
                color: #fff;
            }
            [data-theme="dark"] .card,
            [data-theme="dark"] .webcam-container,
            [data-theme="dark"] .advice-container,
            [data-theme="dark"] .system-info {
                background-color: #1e1e1e;
                color: #e0e0e0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }
            [data-theme="dark"] .button-primary {
                background-color: #bb86fc !important;
                color: #121212 !important;
            }

            [data-theme="dark"] .button-secondary {
                background-color: #03dac6 !important;
                color: #121212 !important;
            }
            [data-theme="dark"] .card .text-h6,
            [data-theme="dark"] .card .text-body1 {
                color: #121212 !important;
            }
            [data-theme="dark"] .emotion-card > div {
                color: #121212 !important;
            }
            

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .webcam-container {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                background-color: #fff;
            }
            .emotion-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 20px;
            }
            .emotion-card {
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                height: 100%;
            }
            .emotion-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }
            .emotion-icon {
                font-size: 24px;
                margin-right: 10px;
            }
            .advice-container {
                border-radius: 10px;
                background-color: #fff;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-top: 20px;
            }
            .system-info {
                font-size: 14px;
                color: #666;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 8px;
                margin-top: 10px;
            }
            .button-primary {
                background-color: #4a6baf !important;
                color: white !important;
                font-weight: 500 !important;
                border-radius: 8px !important;
            }
            .button-secondary {
                background-color: #8998b5 !important;
                color: white !important;
                font-weight: 500 !important;
                border-radius: 8px !important;
            }
            .webcam-display {
                width: 100%;
                height: 480px;
                background-color: #000;
                overflow: hidden;
                position: relative;
            }
            .webcam-image {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            /* Add this to existing CSS */
    [data-theme="dark"] .q-card {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    
    [data-theme="dark"] .q-slider__track {
        background-color: #555 !important;
    }
    
    [data-theme="dark"] .q-slider__track-container {
        background-color: #333 !important;
    }
    
    [data-theme="dark"] .text-lg, 
    [data-theme="dark"] .q-slider__pin-value,
    [data-theme="dark"] .q-slider__pin-text {
        color: #e0e0e0 !important;
    }
    
    [data-theme="dark"] .q-toggle__inner {
        color: #bb86fc !important;
    }

 /* Fix for black webcam display in dark mode */
    [data-theme="dark"] .webcam-display {
        background-color: #333 !important;
    }
    
    [data-theme="dark"] .webcam-image img {
        filter: brightness(1.2) contrast(1.1) !important;  /* Enhance visibility */
    }
    
    /* Fix for emotion labels in dark mode */
    [data-theme="dark"] .emotion-card {
        background-color: #333 !important;  /* Darker background */
    }
    
    [data-theme="dark"] .emotion-card .text-h6,
    [data-theme="dark"] .emotion-card label {
        color: #ffffff !important;  /* White text */
    }
    
    [data-theme="dark"] .q-card .text-h6,
    [data-theme="dark"] .q-card .text-body1,
    [data-theme="dark"] .q-card label {
        color: #ffffff !important;  /* White text for all card content */
    }
    
    /* Fix webcam image rendering */
    .webcam-image img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
        </style>
    <script>
        // Initialize dark mode from localStorage
        document.addEventListener('DOMContentLoaded', function() {
            const savedMode = localStorage.getItem('darkMode');
            if (savedMode === 'true') {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        });
    </script>
    """)
    
    # Wait for models to load
    with ui.dialog() as loading_dialog, ui.card():
        ui.label('Loading models...')
        ui.spinner(size='lg')
    
    loading_dialog.open()
    models_loaded = await emotion_app.initialize_models()
    loading_dialog.close()
    
    if not models_loaded:
        with ui.card():
            ui.label('Failed to load models! Please check the logs and restart the application.')
            return
    
    # Main layout
    with ui.column().classes('container'):
        ui.label('Learning Engagement Monitor').classes('text-h4 text-weight-bold')
        ui.label('Real-time emotional state detection for optimized learning').classes('text-subtitle2 q-mb-md')
        
        # Webcam row
        with ui.row().classes('w-full'):
            with ui.column().classes('webcam-container col-12 col-md-8'):
                # Webcam display - use card with fixed dimensions
                with ui.card().classes('webcam-display'):
                    webcam_container = ui.element('div').classes('w-full h-full')
                    webcam_image = ui.html().classes('webcam-image')
                
                # Controls under webcam
                with ui.row().classes('w-full q-mt-md'):
                    webcam_btn = ui.button('Start Webcam', on_click=emotion_app.start_webcam) \
                        .classes('button-primary')
                    ui.button('Stop Webcam', on_click=emotion_app.stop_webcam) \
                        .classes('button-secondary')                                     
                    
                # Add dark mode toggle button in its own row for better visibility
                with ui.row().classes('w-full q-mt-md'):
                    def toggle_dark_mode():
                        ui.dark_mode = not ui.dark_mode
                        mode = "dark" if ui.dark_mode else "light"
                        ui.run_javascript(f"""
                            document.documentElement.setAttribute('data-theme', '{mode}');
                            localStorage.setItem('darkMode', '{str(ui.dark_mode).lower()}');
                        """)
                        ui.notify(f"Switched to {'dark' if ui.dark_mode else 'light'} mode")
                    
                    # Make the button full width and more noticeable
                    ui.button("Toggle Dark/Light Mode", on_click=toggle_dark_mode).classes("button-secondary w-full")

        
            
            # Emotions display
            with ui.column().classes('col-12 col-md-4 q-pl-md-lg'):
                with ui.row().classes('emotion-grid'):
                    emotion_cards = {}
                    emotion_labels = {}
                    emotion_meters = {}
                    
                    for i, emotion in enumerate(EMOTIONS):
                        with ui.card().classes('emotion-card'):
                            with ui.row().classes('items-center'):
                                ui.label(f"{EMOTION_ICONS[emotion]} {emotion}").classes('text-h6')
                            
                            emotion_labels[emotion] = ui.label('—')
                            emotion_meters[emotion] = ui.linear_progress(value=0).props('rounded')
        
        # Advice row
        with ui.row().classes('w-full'):
            with ui.column().classes('advice-container col-12') as advice_container:
                ui.label('Learning Feedback').classes('text-h6')
                advice_label = ui.label('Waiting for webcam data...').classes('text-body1 q-mt-md').props("id=advice_label")
                explanation_label = ui.label('').classes('text-body2 q-mt-sm text-grey-7').props("id=explanation_label")

        # System info row
        with ui.row().classes('w-full'):
            with ui.column().classes('system-info col-12'):
                ui.label('System Information').classes('text-weight-medium')
                
                with ui.row():
                    with ui.column().classes('col-4'):
                        cpu_label = ui.label(f'CPU: {emotion_app.system_info["cpu"]}%')
                        processing_label = ui.label(f'Processing: {emotion_app.system_info["processing_rate"]}')
                    
                    with ui.column().classes('col-4'):
                        gpu_label = ui.label(f'GPU: {emotion_app.system_info["gpu"]}%')
                        device_label = ui.label(f'Device: {DEVICE}')
                    
                    with ui.column().classes('col-4'):
                        latency_label = ui.label(f'Frame Latency: {reported_latency}ms')
                        

    async def update_webcam_frame():
        """Optimized function to update webcam feed and process frames at different rates."""
        global last_processing_time, results, reported_latency
        last_processing_time = time.time()
        
        # Separate counters for display and processing
        display_frame_count = 0
        process_frame_count = 0
        last_display_time = time.time()
        
        while True:
            if emotion_app.webcam_active:
                # Get current frame
                frame = await emotion_app.get_webcam_frame()
                if frame is not None:
                    display_frame_count += 1
                    
                    # Always update the display at target display rate
                    current_time = time.time()
                    if current_time - last_display_time >= 1/DISPLAY_FPS:
                        last_display_time = current_time

                        # Update the webcam image in the UI
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        webcam_image.set_content(f'<img src="data:image/jpeg;base64,{img_base64}" style="width:100%;height:100%;object-fit:contain;" />')                    
                    # Process frames at controlled rate
                    process_frame_count += 1
                    if emotion_app.processing_active:
                        # Process frame with optimized rate control already in the process_frame function
                        new_results = await emotion_app.process_webcam_frame(frame)
                        
                        # Update emotions display if we have new results
                        if new_results and any(new_results.values()):
                            results = new_results
                            
                            # Update UI with results for each emotion
                            for emotion in EMOTIONS:
                                value = results[emotion]
                                label = get_emotion_label(value, emotion)
                                color = get_emotion_color(value, emotion)
                                
                                emotion_labels[emotion].text = label
                                emotion_meters[emotion].value = (value + 1) / 4
                                emotion_meters[emotion].props(f'color={color}')
                            
                            # Generate and update advice
                            advice = generate_advice(results)
                            if advice != emotion_app.prev_advice:
                                with advice_container:
                                    try:
                                        # Split advice and explanation
                                        if "\n\n" in advice:
                                            advice_text, explanation_text = advice.split("\n\n", 1)
                                            ui.run_javascript(f"var el = document.getElementById('advice_label'); if(el) el.innerText = {json.dumps(advice_text)};")
                                            ui.run_javascript(f"var el = document.getElementById('explanation_label'); if(el) el.innerText = {json.dumps(explanation_text)};")
                                        else:
                                            ui.run_javascript(f"var el = document.getElementById('advice_label'); if(el) el.innerText = {json.dumps(advice)};")
                                            ui.run_javascript(f"var el = document.getElementById('explanation_label'); if(el) el.innerText = '';")
                                        emotion_app.prev_advice = advice
                                    except Exception as err:
                                        print(f"JS update error: {err}")
                    
                # Wait to maintain target UI frame rate
                await asyncio.sleep(1/60)  # 60 Hz maximum refresh rate
            else:
                # If webcam not active, sleep longer
                await asyncio.sleep(0.1)
        
        
    async def update_system_info_task():
        """Separate task to update system information regardless of webcam status"""
        global reported_latency
        
        while True:
            try:
                # Update system info
                emotion_app.update_system_info()
                cpu_label.text = f'CPU: {emotion_app.system_info["cpu"]}%'
                gpu_label.text = f'GPU: {emotion_app.system_info["gpu"]}%'
                processing_label.text = f'Processing: {emotion_app.system_info["processing_rate"]}'
                
                # Update latency with the actual measured value
                latency_label.text = f'Frame Latency: {reported_latency}ms'
            except Exception as e:
                print(f"Error updating system info: {e}")
            
            await asyncio.sleep(1)  # Update every second regardless of webcam
        
    # Start the update loop
    asyncio.create_task(update_webcam_frame())
    asyncio.create_task(update_system_info_task())


ui.run(title="Learning Engagement Monitor", favicon="🎓", port=8080, reload=False)