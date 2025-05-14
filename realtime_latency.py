import os
import sys
import time
import numpy as np
import cv2
import torch
import xgboost as xgb
import psutil
import queue
import asyncio
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from skimage.feature import hog
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

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

# Adjusted gates based on UI feedback and testing
EMOTION_GATES = {
    "Engagement": 0.15,  # Slightly higher for more stability
    "Boredom": 0.40,     # Slightly lower to reduce frequent switches
    "Confusion": 0.20,   # Higher threshold to improve fusion when needed
    "Frustration": 0.25  # Higher threshold to leverage XGBoost more
}

# Fusion strategies for each emotion
FUSION_STRATEGIES = {
    "Engagement": "mobilenet_only",  # Use MobileNet predictions unless confidence extremely low
    "Boredom": "gated_blend",        # Blend probabilities when MobileNet confidence below threshold
    "Confusion": "gated_weighted",    # Use weighted fusion for Confusion if MobileNet confidence low
    "Frustration": "gated_switch"     # Hard switch to XGBoost for very low confidence predictions
}

# Testing configuration for fusion strategies
TEST_FUSION_STRATEGIES = True  # Set to True to test different fusion strategies
FUSION_TEST_MODES = [
    "all_mobilenet", 
    "all_xgboost", 
    "gated_fusion"  # Uses emotion-specific gates and strategies
]
CURRENT_TEST_MODE = "gated_fusion"  # Default to the full gated fusion approach

# Track fusion-specific latency
fusion_latencies = {
    "all_mobilenet": deque(maxlen=100),
    "all_xgboost": deque(maxlen=100),
    "gated_fusion": deque(maxlen=100)
}

# Emotion-specific latency tracking
emotion_latencies = {emotion: deque(maxlen=100) for emotion in EMOTIONS}
# Set CUDA device if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_INFO = f"Using {DEVICE} for inference"

# Neural model config
FRAME_HISTORY = 40      # Number of frames to keep for temporal processing
SMOOTHING_WINDOW = 28   # Number of predictions to smooth over time

# Processing config
BATCH_SIZE = 8          # Process frames in batches for better efficiency
LATENCY_WINDOW = 30     # Time window in seconds for latency statistics

# Display settings
DISPLAY_FPS = 30        # Target display FPS
PROCESS_FPS = 8         # Target processing FPS

# Resource pools
thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 1)

# Latency tracking
latency_history = deque(maxlen=500)  # Store last 500 latency measurements
frame_buffer = deque(maxlen=FRAME_HISTORY)
batch_buffer = []        # Buffer for batch processing
results_buffer = deque(maxlen=SMOOTHING_WINDOW)  # For temporal smoothing

# Track gate triggering statistics
gate_triggered_counts = {emotion: 0 for emotion in EMOTIONS}
gate_total_counts = {emotion: 0 for emotion in EMOTIONS}


# Add a function to display the emotion-specific fusion latencies in the console
def log_emotion_fusion_stats():
    """Log emotion-specific fusion latencies to console."""
    print(f"\n{'-'*70}")
    print(f"EMOTION-SPECIFIC FUSION ANALYSIS")
    print(f"{'-'*70}")
    
    all_emotion_averages = {}
    
    # Calculate averages for each emotion
    for emotion in EMOTIONS:
        if emotion in emotion_latencies and len(emotion_latencies[emotion]) > 0:
            avg_time = sum(emotion_latencies[emotion]) / len(emotion_latencies[emotion])
            all_emotion_averages[emotion] = avg_time
            strategy = FUSION_STRATEGIES[emotion]
            gate = EMOTION_GATES[emotion]
            
            # Calculate how often the gate was triggered
            if emotion in gate_triggered_counts and emotion in gate_total_counts and gate_total_counts[emotion] > 0:
                trigger_percentage = (gate_triggered_counts[emotion] / gate_total_counts[emotion]) * 100
            else:
                trigger_percentage = 0.0
                
            print(f"{emotion:12} | Strategy: {strategy:15} | Gate: {gate:.3f} | Used XGBoost: {trigger_percentage:.1f}% | Latency: {avg_time:.3f} ms")
    
    # Calculate overall average
    if all_emotion_averages:
        overall_avg = sum(all_emotion_averages.values()) / len(all_emotion_averages)
        print(f"{'-'*70}")
        print(f"Overall Fusion Average Latency: {overall_avg:.3f} ms")
    
    print(f"{'-'*70}\n")
    
    
# function to track full end-to-end fusion latency
def log_complete_fusion_stats(latency_tracker, mobilenet_times, xgboost_times):
    """Log complete fusion latency statistics including all components."""
    print(f"\n{'-'*70}")
    print(f"COMPLETE GATED FUSION LATENCY ANALYSIS")
    print(f"{'-'*70}")
    
    emotion_complete_latencies = {}
    gate_activation_rates = {}
    
    # Calculate average component times
    avg_mobilenet = sum(mobilenet_times) / len(mobilenet_times) if mobilenet_times else 0
    avg_xgboost = sum(xgboost_times) / len(xgboost_times) if xgboost_times else 0
    decision_latencies = {}
    
    for emotion in EMOTIONS:
        if emotion in emotion_latencies and len(emotion_latencies[emotion]) > 0:
            # This is just the decision logic time
            decision_time = sum(emotion_latencies[emotion]) / len(emotion_latencies[emotion])
            decision_latencies[emotion] = decision_time
            
            # Calculate how often the gate was triggered
            if emotion in gate_triggered_counts and emotion in gate_total_counts and gate_total_counts[emotion] > 0:
                trigger_rate = gate_triggered_counts[emotion] / gate_total_counts[emotion]
                gate_activation_rates[emotion] = trigger_rate
            else:
                trigger_rate = 0.0
                gate_activation_rates[emotion] = 0
            
            # Calculate the complete latency based on the strategy
            strategy = FUSION_STRATEGIES[emotion]
            
            # For all strategies, we need MobileNet results first
            complete_latency = avg_mobilenet
            
            # Then add additional costs based on strategy and gate triggering
            if strategy == "mobilenet_only":
                # Always just MobileNet cost + minimal decision logic
                complete_latency += decision_time
            elif strategy == "gated_switch":
                # When gate is triggered, add XGBoost cost
                complete_latency += (trigger_rate * avg_xgboost) + decision_time
            elif strategy in ["gated_blend", "gated_weighted"]:
                # These strategies need both models when gate is triggered
                complete_latency += (trigger_rate * avg_xgboost) + decision_time
            
            emotion_complete_latencies[emotion] = complete_latency
    
    # Print out complete fusion latency analysis
    if emotion_complete_latencies:
        # Calculate best, worst, average cases
        latencies = list(emotion_complete_latencies.values())
        best_case = min(latencies)
        worst_case = max(latencies)
        avg_case = sum(latencies) / len(latencies)
        
        print(f"FUSION LATENCY METRICS (including model inference times):")
        print(f"  Best emotion:    {min(emotion_complete_latencies, key=emotion_complete_latencies.get)} ({best_case:.2f} ms)")
        print(f"  Average case:    {avg_case:.2f} ms")
        print(f"  Worst emotion:   {max(emotion_complete_latencies, key=emotion_complete_latencies.get)} ({worst_case:.2f} ms)")
        print(f"  Effective FPS:   {1000/avg_case:.1f}")
        print(f"{'-'*70}")
        
        # Print per-emotion breakdown
        print(f"PER-EMOTION BREAKDOWN:")
        for emotion in EMOTIONS:
            if emotion in emotion_complete_latencies:
                strategy = FUSION_STRATEGIES[emotion]
                gate = EMOTION_GATES[emotion]
                complete_latency = emotion_complete_latencies[emotion]
                trigger_pct = gate_activation_rates[emotion] * 100
                
                print(f"  {emotion:12} | Strategy: {strategy:15} | Gate: {gate:.2f} | XGBoost Used: {trigger_pct:.1f}% | " +
                      f"Latency: {complete_latency:.2f} ms | FPS: {1000/complete_latency:.1f}")
    
    print(f"{'-'*70}")
    print(f"Component times: MobileNet={avg_mobilenet:.2f}ms, XGBoost={avg_xgboost:.2f}ms, Decision={avg_case:.2f}ms")
    print(f"{'-'*70}\n")

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
            
            chunk_feats = self.mobilenet(chunk)
            feats.append(chunk_feats.view(B, end-i, self.feat_dim))
        
        x = torch.cat(feats, dim=1)
        
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(x)
        h_final = torch.cat([h_n[0], h_n[1]], dim=1)
        
        # Get outputs
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
        
        # If ONNX file exists, use it
        if onnx_path.exists():
            print(f"ONNX model found at {onnx_path}. Loading directly.")
            available_providers = ort.get_available_providers()
            print(f"Using ONNX Runtime with available providers: {available_providers}")
            
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
        
        # Create dummy input and export to ONNX
        print(f"Converting model to ONNX format at {onnx_path}")
        dummy_input = torch.randn(1, FRAME_HISTORY, 3, 224, 224, device='cpu')
        
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
#                      INFERENCE FUNCTIONS
# -------------------------------------------------------------------------
async def run_pro_ensemble_batch(frames_batch, model_info):
    """Run ProEnsembleDistillation model on a batch of frame buffers."""
    try:
        if not model_info or 'session' not in model_info or not model_info['session']:
            return {emo: np.array([[0.25, 0.25, 0.25, 0.25]] * len(frames_batch)) for emo in EMOTIONS}
            
        # Process frames in batches
        batch_size = len(frames_batch)
        
        # Stack frames into batch dimension for ONNX
        frames_stacked = np.stack(frames_batch)
        
        # Convert to ONNX input format
        input_data = {'input': frames_stacked.astype(np.float32)}
        
        # Run inference
        results = model_info['session'].run(None, input_data)
        output = results[0]  # Shape: [batch, 4, 4] (batch, emotion, class)
        
        # Convert output to probabilities with softmax for each item in batch
        probs = {}
        for i, emo in enumerate(EMOTIONS):
            emo_logits = output[:, i]  # Shape: [batch, 4]
            emo_probs = np.array([F.softmax(torch.tensor(item), dim=0).numpy() for item in emo_logits])
            probs[emo] = emo_probs
        
        return probs
    except Exception as e:
        print(f"Error in batch ProEnsembleDistillation inference: {e}")
        import traceback
        traceback.print_exc()
        return {emo: np.array([[0.25, 0.25, 0.25, 0.25]] * len(frames_batch)) for emo in EMOTIONS}

async def run_xgboost_batch(hog_features_batch, xgb_models):
    """Run XGBoost models on a batch of HOG features."""
    try:
        batch_size = len(hog_features_batch)
        results = {emo: np.zeros((batch_size, 4)) for emo in EMOTIONS}
        
        # Stack features for batch prediction
        features_stacked = np.vstack(hog_features_batch)
        
        # Create DMatrix for all features
        dmat = xgb.DMatrix(features_stacked)
        
        for emo in EMOTIONS:
            if emo in xgb_models:
                # Get predictions for all samples in batch
                preds = xgb_models[emo].predict(dmat)
                
                # Convert to one-hot encoding
                for i, pred in enumerate(preds):
                    one_hot = np.zeros(4)
                    one_hot[int(pred)] = 1.0
                    results[emo][i] = one_hot
            else:
                # Default uniform distribution if model is missing
                results[emo] = np.ones((batch_size, 4)) * 0.25
                
        return results
    except Exception as e:
        print(f"Error in batch XGBoost inference: {e}")
        return {emo: np.ones((batch_size, 4)) * 0.25 for emo in EMOTIONS}

def select_final_prediction(mobilenet_probs, xgboost_probs, emotion):
    """
    Apply emotion-specific fusion strategies based on benchmark analysis.
    
    Note: This function only measures the DECISION LOGIC latency, not the full
    model inference times. The actual model inference times are measured separately
    in the process_batch function (mobilenet_time and xgboost_time variables).
    
    Args:
        mobilenet_probs: Probability distribution from MobileNet model
        xgboost_probs: Probability distribution from XGBoost model
        emotion: Emotion type being processed
    
    Returns:
        Predicted class after fusion, fusion decision latency in ms, and whether gate was triggered
    """
    # Start fusion timing (DECISION LOGIC ONLY)
    fusion_start = time.time()
    
    # Get maximum probability and its class from MobileNet
    mobilenet_conf = np.max(mobilenet_probs)
    mobilenet_class = np.argmax(mobilenet_probs)
    
    # Get maximum probability and its class from XGBoost
    xgboost_conf = np.max(xgboost_probs)
    xgboost_class = np.argmax(xgboost_probs)
    
    # Test different fusion strategies based on configuration
    gate_triggered = False
    
    if TEST_FUSION_STRATEGIES:
        if CURRENT_TEST_MODE == "all_mobilenet":
            # Always use MobileNet prediction
            fusion_time = (time.time() - fusion_start) * 1000
            return mobilenet_class, fusion_time, gate_triggered
            
        elif CURRENT_TEST_MODE == "all_xgboost":
            # Always use XGBoost prediction
            fusion_time = (time.time() - fusion_start) * 1000
            return xgboost_class, fusion_time, gate_triggered
    
    # Default to gated fusion (emotion-specific)
    gate = EMOTION_GATES[emotion]
    strategy = FUSION_STRATEGIES[emotion]
    
    # Apply the appropriate fusion strategy
    if mobilenet_conf >= gate:
        # If MobileNet confidence is above gate, use its prediction
        result = mobilenet_class
    else:
        # Gate was triggered - using alternative strategy
        gate_triggered = True
        
        # Apply emotion-specific fusion strategy
        if strategy == "mobilenet_only":
            # For "mobilenet_only", still use MobileNet even below gate
            result = mobilenet_class
            
        elif strategy == "gated_blend":
            # For "gated_blend", blend probabilities with 60% MobileNet, 40% XGBoost
            blended_probs = 0.6 * mobilenet_probs + 0.4 * xgboost_probs
            result = np.argmax(blended_probs)
            
        elif strategy == "gated_weighted":
            # For "gated_weighted", use weighted fusion approach for Confusion
            weighted_probs = 0.3 * mobilenet_probs + 0.7 * xgboost_probs
            result = np.argmax(weighted_probs)
            
        elif strategy == "gated_switch":
            # For "gated_switch", hard switch to XGBoost for very low confidence
            result = xgboost_class
        else:
            # Fallback to MobileNet prediction if strategy not recognized
            result = mobilenet_class
    
    # Calculate fusion time
    fusion_time = (time.time() - fusion_start) * 1000
    return result, fusion_time, gate_triggered

def apply_temporal_smoothing(current_pred, emotion, results_buffer):
    """Apply temporal smoothing to predictions for stability."""
    # Add current prediction to the buffer
    if len(results_buffer) == 0:
        # Initialize with the current prediction for all positions
        results_buffer.extend([{emotion: current_pred for emotion in EMOTIONS}] * SMOOTHING_WINDOW)
    else:
        results_buffer.append({emotion: current_pred for emotion in EMOTIONS})
    
    # Apply exponential weighting (more recent predictions have higher weight)
    weights = np.exp(np.linspace(0, 1, len(results_buffer)))
    weights /= weights.sum()
    
    # Calculate weighted average for the emotion
    counts = np.zeros(4)
    for i, result in enumerate(results_buffer):
        pred = result[emotion]
        counts[pred] += weights[i]
    
    # Return the class with highest smoothed count
    return np.argmax(counts)

# -------------------------------------------------------------------------
#                      LATENCY MEASUREMENT
# -------------------------------------------------------------------------
class LatencyTracker:
    def __init__(self):
        self.latency_history = deque(maxlen=500)  # Last 500 latency measurements
        self.capture_times = deque(maxlen=100)    # Last 100 capture times
        self.preprocess_times = deque(maxlen=100) # Last 100 preprocess times
        self.hog_times = deque(maxlen=100)        # Last 100 HOG extraction times
        self.mobilenet_times = deque(maxlen=100)  # Last 100 MobileNet inference times
        self.xgboost_times = deque(maxlen=100)    # Last 100 XGBoost inference times
        self.postprocess_times = deque(maxlen=100)# Last 100 post-processing times
        self.total_times = deque(maxlen=100)      # Last 100 total processing times
        
        # Time window tracking
        self.window_start_time = time.time()
        self.window_latencies = []  # Latencies within current time window
        self.window_duration = LATENCY_WINDOW  # Duration of time window in seconds
        self.window_history = []    # List of window statistics
        
    def add_latency(self, total_ms, capture_ms=None, preprocess_ms=None, 
                hog_ms=None, mobilenet_ms=None, xgboost_ms=None, 
                postprocess_ms=None):
        """Add a new latency measurement in milliseconds."""
        self.latency_history.append(total_ms)
        if capture_ms is not None:
            self.capture_times.append(capture_ms)
        if preprocess_ms is not None:
            self.preprocess_times.append(preprocess_ms)
        if hog_ms is not None:
            self.hog_times.append(hog_ms)
        if mobilenet_ms is not None:
            self.mobilenet_times.append(mobilenet_ms)
        if xgboost_ms is not None:
            self.xgboost_times.append(xgboost_ms)
        if postprocess_ms is not None:
            self.postprocess_times.append(postprocess_ms)
        if total_ms is not None:
            self.total_times.append(total_ms)
            
        # Add to window tracking
        self.window_latencies.append(total_ms)
        
        # Check if window duration has elapsed
        current_time = time.time()
        elapsed = current_time - self.window_start_time
        
        if elapsed >= self.window_duration and self.window_latencies:
            # Calculate window statistics
            window_stats = {
                'timestamp': current_time,
                'duration': elapsed,
                'count': len(self.window_latencies),
                'min': min(self.window_latencies),
                'max': max(self.window_latencies),
                'avg': sum(self.window_latencies) / len(self.window_latencies),
                'p95': np.percentile(self.window_latencies, 95) if len(self.window_latencies) > 5 else None
            }
            
            # Save window statistics
            self.window_history.append(window_stats)
            
            # Log window statistics
            print(f"\n{'-'*50}")
            print(f"LATENCY REPORT FOR {elapsed:.1f}s WINDOW:")
            print(f"  Best case:    {window_stats['min']:.2f} ms")
            print(f"  Average case: {window_stats['avg']:.2f} ms")
            print(f"  Worst case:   {window_stats['max']:.2f} ms")
            print(f"  P95:          {window_stats['p95']:.2f} ms" if window_stats['p95'] else "  P95:          N/A")
            print(f"  Processed:    {window_stats['count']} frames")
            print(f"{'-'*50}\n")
            
            # Also log emotion-specific fusion stats with full component times
            if TEST_FUSION_STRATEGIES and CURRENT_TEST_MODE == "gated_fusion":
                log_emotion_fusion_stats()  # Original report
                log_complete_fusion_stats(self, self.mobilenet_times, self.xgboost_times)  # New complete report
            
            # Reset for next window
            self.window_start_time = current_time
            self.window_latencies = []
            
    def add_fusion_latency(self, strategy, time_ms, emotion=None):
        """Track latency for specific fusion strategy."""
        global fusion_latencies, emotion_latencies
        
        if strategy in fusion_latencies:
            fusion_latencies[strategy].append(time_ms)
        
        if emotion and emotion in emotion_latencies:
            emotion_latencies[emotion].append(time_ms)
    
    def get_stats(self):
        """Get summary statistics of latency measurements."""
        if not self.latency_history:
            return {
                'best': 0,
                'worst': 0,
                'average': 0,
                'current': 0,
                'percentile_95': 0,
                'component_averages': {
                    'capture': 0,
                    'preprocess': 0,
                    'hog': 0,
                    'mobilenet': 0,
                    'xgboost': 0,
                    'postprocess': 0,
                    'total': 0
                },
                'window': {
                    'elapsed': 0,
                    'count': 0,
                    'min': 0,
                    'max': 0,
                    'avg': 0
                }
            }
        
        # Calculate elapsed time in current window
        current_window_elapsed = time.time() - self.window_start_time
        
        stats = {
            'best': min(self.latency_history),
            'worst': max(self.latency_history),
            'average': sum(self.latency_history) / len(self.latency_history),
            'current': self.latency_history[-1],
            'percentile_95': np.percentile(self.latency_history, 95),
            'component_averages': {
                'capture': sum(self.capture_times) / len(self.capture_times) if self.capture_times else 0,
                'preprocess': sum(self.preprocess_times) / len(self.preprocess_times) if self.preprocess_times else 0,
                'hog': sum(self.hog_times) / len(self.hog_times) if self.hog_times else 0,
                'mobilenet': sum(self.mobilenet_times) / len(self.mobilenet_times) if self.mobilenet_times else 0,
                'xgboost': sum(self.xgboost_times) / len(self.xgboost_times) if self.xgboost_times else 0,
                'postprocess': sum(self.postprocess_times) / len(self.postprocess_times) if self.postprocess_times else 0,
                'total': sum(self.total_times) / len(self.total_times) if self.total_times else 0
            },
            'window': {
                'elapsed': current_window_elapsed,
                'count': len(self.window_latencies),
                'min': min(self.window_latencies) if self.window_latencies else 0,
                'max': max(self.window_latencies) if self.window_latencies else 0,
                'avg': sum(self.window_latencies) / len(self.window_latencies) if self.window_latencies else 0
            }
        }
        return stats

# -------------------------------------------------------------------------
#                      BATCH PROCESSING FUNCTIONS
# -------------------------------------------------------------------------
class FrameBatch:
    def __init__(self, max_size=BATCH_SIZE):
        self.frames = []
        self.preprocessed_frames = []
        self.hog_features = []
        self.timestamps = []
        self.max_size = max_size
        
    def add_frame(self, frame, timestamp):
        """Add a new frame to the batch."""
        preprocessed = preprocess_frame(frame)
        hog_feature = extract_hog_features(frame)
        
        self.frames.append(frame)
        self.preprocessed_frames.append(preprocessed)
        self.hog_features.append(hog_feature)
        self.timestamps.append(timestamp)
        
    def is_full(self):
        """Check if the batch is full."""
        return len(self.frames) >= self.max_size
        
    def size(self):
        """Get the current size of the batch."""
        return len(self.frames)
        
    def clear(self):
        """Clear the batch."""
        self.frames = []
        self.preprocessed_frames = []
        self.hog_features = []
        self.timestamps = []

# -------------------------------------------------------------------------
#                      MAIN PROCESSING FUNCTION
# -------------------------------------------------------------------------
async def process_batch(frame_batch, frame_buffer, models, latency_tracker):
    """Process a batch of frames and measure latency."""
    batch_size = frame_batch.size()
    times_per_frame = [{} for _ in range(batch_size)]
    
    if batch_size == 0:
        return [], []
    
    # Record batch start time
    batch_start = time.time()
    
    # For each frame in batch, update frame buffer
    frame_buffers = []
    for i in range(batch_size):
        # Create a copy of the current global frame buffer
        current_buffer = frame_buffer.copy()
        
        # Add this frame to the buffer
        current_buffer.append(frame_batch.preprocessed_frames[i])
        
        # Keep the buffer size fixed
        while len(current_buffer) > FRAME_HISTORY:
            current_buffer.popleft()
            
        # Store this buffer for later
        frame_buffers.append(current_buffer)
    
    # Run MobileNet+LSTM inference on all frame buffers in batch
    mobilenet_start = time.time()
    
    # Prepare full frame history for each frame in batch
    frame_batches = []
    for buffer in frame_buffers:
        # Skip if we don't have enough frames
        if len(buffer) < FRAME_HISTORY:
            # Add placeholder (zeros) so we maintain batch size
            frame_batches.append(np.zeros((FRAME_HISTORY, 3, 224, 224)))
        else:
            frame_batches.append(np.stack(buffer))
    
    # Run batch inference
    pro_results = await run_pro_ensemble_batch(frame_batches, models['student'])
    mobilenet_time = (time.time() - mobilenet_start) * 1000
    
    # Run XGBoost inference
    xgboost_start = time.time()
    xgboost_results = await run_xgboost_batch(frame_batch.hog_features, models['xgboost'])
    xgboost_time = (time.time() - xgboost_start) * 1000
    
    # Apply fusion and post-processing
    postprocess_start = time.time()
    final_results = []
    fusion_times = {emotion: [] for emotion in EMOTIONS}

    for i in range(batch_size):
        frame_result = {}
        
        for emotion in EMOTIONS:
            # Get probabilities from models
            mobilenet_probs = pro_results[emotion][i]
            xgboost_probs = xgboost_results[emotion][i]
            
            # In the process_batch function where you call select_final_prediction:
            raw_pred, fusion_time, gate_triggered = select_final_prediction(mobilenet_probs, xgboost_probs, emotion)

            # Update gate statistics
            gate_total_counts[emotion] = gate_total_counts.get(emotion, 0) + 1
            if gate_triggered:
                gate_triggered_counts[emotion] = gate_triggered_counts.get(emotion, 0) + 1
            fusion_times[emotion].append(fusion_time)
            
            # Update fusion latency tracking
            latency_tracker.add_fusion_latency(CURRENT_TEST_MODE, fusion_time, emotion)
            
            # Apply temporal smoothing
            smooth_pred = apply_temporal_smoothing(raw_pred, emotion, results_buffer)
            
            frame_result[emotion] = smooth_pred
        
        final_results.append(frame_result)
    
    postprocess_time = (time.time() - postprocess_start) * 1000
    
    # Calculate batch total time and per-frame times
    total_batch_time = (time.time() - batch_start) * 1000
    
    # Calculate per-frame processing times
    for i in range(batch_size):
        times = times_per_frame[i]
        times['capture'] = 0  # Set outside this function
        times['preprocess'] = 0  # Already done during batch collection
        times['mobilenet'] = mobilenet_time / batch_size
        times['xgboost'] = xgboost_time / batch_size
        times['postprocess'] = postprocess_time / batch_size
        times['total'] = total_batch_time / batch_size
        
        # Update latency tracker with this frame's times
        latency_tracker.add_latency(
            times['total'],
            capture_ms=times.get('capture', 0),
            preprocess_ms=times.get('preprocess', 0),
            hog_ms=0,  # Already counted in preprocessing
            mobilenet_ms=times.get('mobilenet', 0),
            xgboost_ms=times.get('xgboost', 0),
            postprocess_ms=times.get('postprocess', 0)
        )
    
    return final_results, times_per_frame

# -------------------------------------------------------------------------
#                      VISUALIZATION FUNCTIONS
# -------------------------------------------------------------------------
def draw_latency_stats(frame, stats):
    """Draw minimal latency statistics on frame."""
    frame_height, frame_width = frame.shape[:2]
    
    # Create a semi-transparent overlay instead of solid background
    overlay = frame.copy()
    
    # Draw a small stats box in the TOP-LEFT corner (changed from top-right)
    stats_width = 220
    stats_height = 80
    top_left_x = 10  # Changed from right side to left side
    
    cv2.rectangle(overlay, (top_left_x, 10), (top_left_x + stats_width, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Show just the key metrics
    cv2.putText(frame, f"Latency (ms)", (top_left_x + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Best: {stats['best']:.1f} | Avg: {stats['average']:.1f}", (top_left_x + 10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Current: {stats['current']:.1f} | Worst: {stats['worst']:.1f}", (top_left_x + 10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_fusion_latencies(frame, stats):
    """Draw complete fusion latency statistics on frame."""
    frame_height, frame_width = frame.shape[:2]
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    
    # Calculate full gated fusion latencies
    emotion_complete_latencies = {}
    avg_mobilenet = stats['component_averages']['mobilenet']
    avg_xgboost = stats['component_averages']['xgboost']
    
    for emotion in EMOTIONS:
        if emotion in emotion_latencies and len(emotion_latencies[emotion]) > 0:
            decision_time = sum(emotion_latencies[emotion]) / len(emotion_latencies[emotion])
            
            # Calculate gate trigger rate
            if emotion in gate_triggered_counts and emotion in gate_total_counts and gate_total_counts[emotion] > 0:
                trigger_rate = gate_triggered_counts[emotion] / gate_total_counts[emotion]
            else:
                trigger_rate = 0.0
            
            # Calculate complete latency based on strategy
            strategy = FUSION_STRATEGIES[emotion]
            complete_latency = avg_mobilenet
            
            if strategy != "mobilenet_only" and trigger_rate > 0:
                complete_latency += trigger_rate * avg_xgboost
            
            emotion_complete_latencies[emotion] = complete_latency
    
    # Draw a box at the bottom of the screen
    stats_height = 40 + len(EMOTIONS) * 20
    bottom_y = frame_height - stats_height - 10
    
    cv2.rectangle(overlay, (10, bottom_y), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Show fusion mode and title
    cv2.putText(frame, f"Full Gated Fusion Analysis", (20, bottom_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add emotion-specific info
    y_pos = bottom_y + 45
    for emotion in EMOTIONS:
        if emotion in emotion_complete_latencies:
            complete_latency = emotion_complete_latencies[emotion]
            trigger_pct = (gate_triggered_counts.get(emotion, 0) / max(1, gate_total_counts.get(emotion, 1))) * 100
            fps = 1000 / complete_latency if complete_latency > 0 else 0
            
            cv2.putText(frame, 
                       f"{emotion[:4]}: {complete_latency:.1f}ms | XGB: {trigger_pct:.1f}% | FPS: {fps:.1f}", 
                       (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_pos += 20
    
    # Show average
    if emotion_complete_latencies:
        avg_latency = sum(emotion_complete_latencies.values()) / len(emotion_complete_latencies)
        avg_fps = 1000 / avg_latency if avg_latency > 0 else 0
        cv2.putText(frame, f"Overall: {avg_latency:.1f}ms | FPS: {avg_fps:.1f}", 
                   (20, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
# -------------------------------------------------------------------------
#                      MAIN APPLICATION
# -------------------------------------------------------------------------
async def main():
    global CURRENT_TEST_MODE, fusion_latencies, emotion_latencies
    
    # Initialize models
    print("Loading models...")
    models = {
        'student': None,
        'xgboost': None
    }
    
    # Load models
    models['xgboost'] = load_xgboost_models()
    models['student'] = load_student_model()
    
    if not models['student'] and not models['xgboost']:
        print("Error: Failed to load any models!")
        return
    
    print("Models loaded successfully!")
    
    # Initialize webcam
    print("Starting webcam...")
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
    
    if not webcam.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Set webcam properties for better performance
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize frame buffer and latency tracker
    frame_buffer = deque(maxlen=FRAME_HISTORY)
    latency_tracker = LatencyTracker()
    current_batch = FrameBatch(max_size=BATCH_SIZE)
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Results storage
    last_results = {emo: 0 for emo in EMOTIONS}
    
    print("Starting latency measurement...")
    
    while True:
        # Record frame capture start time
        capture_start = time.time()
        
        # Read frame from webcam
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to read from webcam!")
            break
        
        # Calculate capture latency
        capture_time = (time.time() - capture_start) * 1000
        
        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add frame to current batch
        current_batch.add_frame(frame, time.time())
        
        # Process batch if it's full or process incomplete batch periodically
        if current_batch.is_full():
            # Process the batch
            batch_results, batch_times = await process_batch(current_batch, frame_buffer, models, latency_tracker)
            
            # Update frame buffer with newest frame for next iteration
            if current_batch.preprocessed_frames:
                for preprocessed in current_batch.preprocessed_frames:
                    frame_buffer.append(preprocessed)
                    # Keep frame_buffer at max size
                    while len(frame_buffer) > FRAME_HISTORY:
                        frame_buffer.popleft()
            
            # Update last results with most recent prediction
            if batch_results:
                last_results = batch_results[-1]
            
            # Clear the batch
            current_batch.clear()
        
        # Get latency statistics
        stats = latency_tracker.get_stats()
        
        # Calculate FPS
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Draw latency statistics on frame
        draw_latency_stats(frame, stats)
        
        # Draw fusion latencies
        draw_fusion_latencies(frame, stats)
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show CPU & GPU usage if available
        cpu_usage = psutil.cpu_percent()
        cv2.putText(frame, f"CPU: {cpu_usage:.1f}%", (frame.shape[1] - 120, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show batch size 
        cv2.putText(frame, f"Batch: {BATCH_SIZE}", (frame.shape[1] - 120, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if torch.cuda.is_available():
            try:
                # Simple approach for NVIDIA GPUs - not precise but gives an indication
                gpu_usage = int(os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits').read().strip())
                cv2.putText(frame, f"GPU: {gpu_usage}%", (frame.shape[1] - 120, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            except:
                pass
        
        # Display predictions
        y_pos = 210
        for emotion in EMOTIONS:
            value = last_results.get(emotion, 0)
            # Get label mapping for emotions
            labels = {
                "Engagement": ["Disengaged", "Low Engagement", "Engaged", "Very Engaged"],
                "Boredom": ["Not Bored", "Slightly Bored", "Bored", "Very Bored"],
                "Confusion": ["Not Confused", "Slightly Confused", "Confused", "Very Confused"],
                "Frustration": ["Not Frustrated", "Slightly Frustrated", "Frustrated", "Very Frustrated"]
            }
            label = labels[emotion][value]
            
            cv2.putText(frame, f"{emotion}: {label}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        
        # Display the resulting frame
        cv2.imshow('Real-time Latency Analysis', frame)
        
        # Print current latency to console periodically
        if fps_frame_count == 0:
            print(f"Latency: {stats['best']:.2f} ms (best), {stats['average']:.2f} ms (avg), {stats['worst']:.2f} ms (worst)")
            
        # Print fusion latencies to console
        if fps_frame_count == 0:
            print(f"Latency: {stats['best']:.2f} ms (best), {stats['average']:.2f} ms (avg), {stats['worst']:.2f} ms (worst)")
            
            # Add fusion stats
            if CURRENT_TEST_MODE in fusion_latencies and len(fusion_latencies[CURRENT_TEST_MODE]) > 0:
                avg_fusion = sum(fusion_latencies[CURRENT_TEST_MODE]) / len(fusion_latencies[CURRENT_TEST_MODE])
                print(f"Fusion ({CURRENT_TEST_MODE}): {avg_fusion:.3f} ms")
                
                # Emotion-specific stats
                print("Emotion-Specific Fusion Latencies:")
                for emotion in EMOTIONS:
                    if emotion in emotion_latencies and len(emotion_latencies[emotion]) > 0:
                        avg_time = sum(emotion_latencies[emotion]) / len(emotion_latencies[emotion])
                        print(f"  {emotion}: {avg_time:.3f} ms ({FUSION_STRATEGIES[emotion]})")
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Add these keyboard shortcuts to main() function after the existing key handlers
        if cv2.waitKey(1) & 0xFF == ord('1'):
            CURRENT_TEST_MODE = "all_mobilenet"
            print(f"Switched to test mode: {CURRENT_TEST_MODE}")
        elif cv2.waitKey(1) & 0xFF == ord('2'):
            CURRENT_TEST_MODE = "all_xgboost"
            print(f"Switched to test mode: {CURRENT_TEST_MODE}")
        elif cv2.waitKey(1) & 0xFF == ord('3'):
            CURRENT_TEST_MODE = "gated_fusion"
            print(f"Switched to test mode: {CURRENT_TEST_MODE}")
        
        # Cap the display framerate
        await asyncio.sleep(1/DISPLAY_FPS)
    
    # Clean up
    webcam.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    print("Starting Real-time Latency Measurement Tool")
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Latency window: {LATENCY_WINDOW} seconds")
    print(f"Testing fusion strategies: {TEST_FUSION_STRATEGIES}")
    print(f"Current fusion mode: {CURRENT_TEST_MODE}")
    print(f"Emotion gates: {EMOTION_GATES}")
    print(f"Fusion strategies: {FUSION_STRATEGIES}")
    asyncio.run(main())