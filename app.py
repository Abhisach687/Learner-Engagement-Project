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

async def run_pro_ensemble(frame_buffer, model_info):
    """Run ProEnsembleDistillation model on a buffer of frames."""
    try:
        # If no model info or session, return default probabilities
        if not model_info or 'session' not in model_info or not model_info['session']:
            return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}
            
        # Process frames in batches for the temporal model
        frames = np.array(frame_buffer)
        
        # Add batch dimension if missing
        if len(frames.shape) == 4:  # [frames, channels, height, width]
            frames = frames.reshape(1, *frames.shape)  # [1, frames, channels, height, width]
        
        # Convert to ONNX input format
        input_data = {'input': frames.astype(np.float32)}
        
        # Run inference
        results = model_info['session'].run(None, input_data)
        output = results[0]  # Shape: [1, 4, 4] (batch, emotion, class)
        
        # Convert output to probabilities with softmax
        probs = {}
        for i, emo in enumerate(EMOTIONS):
            emo_logits = output[0, i]
            emo_probs = F.softmax(torch.tensor(emo_logits), dim=0).numpy()
            probs[emo] = emo_probs
        
        return probs
    except Exception as e:
        print(f"Error in ProEnsembleDistillation inference: {e}")
        import traceback
        traceback.print_exc()
        return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}

async def run_xgboost(hog_feature, xgb_models):
    """Run XGBoost models on HOG features."""
    try:
        results = {}
        # Create DMatrix once for all models
        dmat = xgb.DMatrix(hog_feature.reshape(1, -1))
        
        for emo in EMOTIONS:
            if emo in xgb_models:
                # Get probabilities from model
                raw_pred = xgb_models[emo].predict(dmat, output_margin=True)[0]
                
                # Convert to one-hot encoding for now
                pred_class = int(xgb_models[emo].predict(dmat)[0])
                probs = np.zeros(4)
                probs[pred_class] = 1.0
                
                results[emo] = probs
            else:
                results[emo] = np.array([0.25, 0.25, 0.25, 0.25])
                
        return results
    except Exception as e:
        print(f"Error in XGBoost inference: {e}")
        return {emo: np.array([0.25, 0.25, 0.25, 0.25]) for emo in EMOTIONS}

def select_final_prediction(pro_ensemble_result, xgboost_result, emotion):
    """Select the best model for each emotion based on benchmark performance."""
    if emotion == "Boredom" or emotion == "Confusion" or emotion == "Frustration":
        return xgboost_result  # XGBOOST_HOG outperforms in these emotions
    else:  # emotion == "Engagement"
        return pro_ensemble_result  # ProEnsembleDistillation slightly better here

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


# -------------------------------------------------------------------------
#                      MAIN PROCESSING FUNCTION
# -------------------------------------------------------------------------

async def process_frame(frame, frame_count, frame_buffer, models):
    """Main processing function for a single frame."""
    processing_rate = determine_processing_rate()
    
    # Based on load, we might process only some frames
    if (processing_rate == "process_every_second_frame" and frame_count % 2 != 0) or \
       (processing_rate == "process_every_third_frame" and frame_count % 3 != 0):
        # Return the last result if we're skipping this frame
        if RESULTS_BUFFER:
            return RESULTS_BUFFER[-1]
        else:
            return {emo: 0 for emo in EMOTIONS}
    
    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame.copy())
    frame_buffer.append(preprocessed_frame)
    
    # Keep the buffer size fixed
    while len(frame_buffer) > FRAME_HISTORY:
        frame_buffer.pop(0)
    
    # If we don't have enough frames yet, return placeholder
    if len(frame_buffer) < FRAME_HISTORY:
        return {emo: 0 for emo in EMOTIONS}
    
    # Extract HOG features in parallel thread
    hog_future = asyncio.get_event_loop().run_in_executor(
        thread_pool, extract_hog_features, frame
    )
    
    # Start model inferences in parallel
    pro_ensemble_task = asyncio.create_task(
        run_pro_ensemble(np.stack(frame_buffer), models['student'])
    )
    
    # Wait for HOG extraction to complete
    hog_features = await hog_future
    
    # Run XGBoost inference
    xgboost_task = asyncio.create_task(
        run_xgboost(hog_features, models['xgboost'])
    )
    
    # Gather results and apply model-specific post-processing
    pro_result = await pro_ensemble_task
    xgboost_result = await xgboost_task
    
    final_results = {}
    
    # Process each emotion
    for emotion in EMOTIONS:
        # Get probabilities from models
        pro_prob = pro_result[emotion]
        xgb_prob = xgboost_result[emotion]
        
        # Get class with highest probability for each model
        pro_class = np.argmax(pro_prob)
        xgb_class = np.argmax(xgb_prob)
        
        # Lower the confidence threshold to allow more variability
        if pro_prob[pro_class] > 0.35:  # Lowered from 0.45 to allow more variability
            # Use the predicted class directly for high confidence predictions
            final_results[emotion] = pro_class
        else:
            # For lower confidence, still go through post-processing
            pro_probs = np.array([pro_result[emotion]])
            xgb_probs = np.array([xgboost_result[emotion]])
            
            pro_pred = apply_pro_ensemble_postprocessing(pro_probs, emotion)[0]
            xgb_pred = apply_xgboost_postprocessing(np.array([xgb_class]), emotion)[0]
            
            # Select best model for this emotion
            raw_pred = select_final_prediction(pro_pred, xgb_pred, emotion)
            
            # Apply temporal smoothing
            smooth_pred = apply_temporal_smoothing(raw_pred, emotion)
            
            final_results[emotion] = smooth_pred
    
    # Debug output to see what values are being produced
    print(f"Final results: {final_results}")
    
    return final_results

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
    """Generate educational advice based on emotion detection."""
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
    
    if engagement <= 1:  # Disengaged or Low Engagement
        if boredom >= 2:  # Bored or Very Bored
            return "You seem disengaged. Try taking a short break or switching to a more interactive section."
        if confusion >= 2:  # Confused or Very Confused
            return "You appear to be struggling with this material. Consider reviewing the prerequisites or asking for help."
        if frustration >= 2:  # Frustrated or Very Frustrated
            return "You seem frustrated. Take a moment to relax and break the problem into smaller steps."
        return "Try to engage more actively with the material by taking notes or working through examples."
        
    elif confusion >= 2:  # Engaged but Confused
        return "You're engaged but seem confused. Don't hesitate to ask questions or review previous sections."
        
    elif frustration >= 2:  # Engaged but Frustrated
        return "You're working hard but seem frustrated. Consider a different approach or ask for assistance."
        
    elif boredom >= 2:  # Engaged but Bored
        return "You're attentive but might benefit from more challenging material or interactive exercises."
        
    else:  # Fully engaged, not bored, confused or frustrated
        return "You're demonstrating excellent engagement! Keep up the good work."

# -------------------------------------------------------------------------
#                      MAIN APPLICATION
# -------------------------------------------------------------------------
class EmotionDetectionApp:
    def __init__(self):
        self.frame_count = 0
        self.frame_buffer = []
        self.webcam = None
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
        """Process a single frame from the webcam."""
        if not self.processing_active:
            return self.last_results
                
        self.frame_count += 1
        
        try:
            results = await process_frame(
                frame, 
                self.frame_count,
                self.frame_buffer,
                self.models
            )
            
            # Print results periodically (every 30 frames)
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count} results: {results}")
                print(f"Frame buffer size: {len(self.frame_buffer)}/{FRAME_HISTORY}")
                
                # If we're getting zeros consistently, print more information
                if not any(results.values()):
                    print("WARNING: All zero values in results - check model outputs")
            
            # Always update last_results regardless of values
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
        """Stop the webcam feed."""
        if not self.webcam_active:
            return
            
        self.webcam_active = False
        self.processing_active = False
        
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
        </style>
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
            with ui.column().classes('advice-container col-12'):
                ui.label('Learning Feedback').classes('text-h6')
                advice_label = ui.label('Waiting for webcam data...').classes('text-body1 q-mt-md')
        
        # System info row
        with ui.row().classes('w-full'):
            with ui.column().classes('system-info col-12'):
                ui.label('System Information').classes('text-weight-medium')
                
                with ui.row():
                    with ui.column().classes('col-6'):
                        cpu_label = ui.label(f'CPU: {emotion_app.system_info["cpu"]}%')
                        processing_label = ui.label(f'Processing: {emotion_app.system_info["processing_rate"]}')
                    
                    with ui.column().classes('col-6'):
                        gpu_label = ui.label(f'GPU: {emotion_app.system_info["gpu"]}%')
                        device_label = ui.label(f'Device: {DEVICE}')
    
    async def update_webcam_frame():
        while True:
            if emotion_app.webcam_active:
                frame = await emotion_app.get_webcam_frame()
                if frame is not None:
                    # Process frame
                    results = await emotion_app.process_webcam_frame(frame)
                    
                    # Debug output to identify problems
                    print(f"Debug: Sending to advice generator: {results}")
                    advice = generate_advice(results)
                    print(f"Debug: Generated advice: '{advice}'")
                    
                    # Update UI with results
                    for emotion in EMOTIONS:
                        value = results[emotion]
                        label = get_emotion_label(value, emotion)
                        color = get_emotion_color(value, emotion)
                        
                        emotion_labels[emotion].text = label
                        emotion_meters[emotion].value = (value + 1) / 4
                        emotion_meters[emotion].props(f'color={color}')
                    
                    # FIXED: Single, reliable approach to update advice
                    if any(v > 0 for v in results.values()):
                        # Force a complete UI refresh using a two-step approach
                        advice_label.text = ""  # Clear first to ensure change is detected
                        await asyncio.sleep(0.02)  # Small delay to ensure UI cycle
                        advice_label.text = advice  # Set new text
                    elif advice_label.text == "Waiting for webcam data...":
                        # Only update if still showing initial message
                        advice_label.text = advice
                    
                    # Update system info
                    emotion_app.update_system_info()
                    cpu_label.text = f'CPU: {emotion_app.system_info["cpu"]}%'
                    gpu_label.text = f'GPU: {emotion_app.system_info["gpu"]}%'
                    processing_label.text = f'Processing: {emotion_app.system_info["processing_rate"]}'
                    
                    # Process frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw emotion states
                    y_pos = 30
                    frame_height, frame_width = frame_rgb.shape[:2]
                    
                    for emotion in EMOTIONS:
                        value = results[emotion]
                        label = get_emotion_label(value, emotion)
                        
                        icon_text = {
                            "Engagement": "1",
                            "Boredom": "2", 
                            "Confusion": "3",
                            "Frustration": "4"
                        }[emotion]
                        
                        text = f"{icon_text}: {emotion} - {label}"
                        
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame_rgb, 
                                    (10, y_pos - 15), 
                                    (min(frame_width - 10, 10 + text_size[0] + 10), y_pos + 5),
                                    (240, 240, 240), -1)
                        
                        cv2.putText(frame_rgb, text, (15, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                        y_pos += 25
                    
                    # Convert to JPEG and update display
                    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    webcam_image.set_content(f'<img src="data:image/jpeg;base64,{img_base64}" style="width:100%;height:100%;object-fit:contain;background-color:black;">')
                
            await asyncio.sleep(1/30)  # Target 30 FPS UI updates
        
    # Start the update loop
    asyncio.create_task(update_webcam_frame())

ui.run(title="Learning Engagement Monitor", favicon="🎓", port=8080)