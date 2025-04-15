import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import from your main script
from ProEnsembleDistillation import (
    MobileNetV2LSTMStudent, DAiSEERawDataset, Distiller, DAiSEEEnsemble,  # Added DAiSEEEnsemble
    EMOTIONS, DEVICE, DISTILL_MODELS_DIR, METRICS_DIR, VISUALS_DIR, LOG_FILE,
    LOG_DIR, CACHE_DIR, FRAMES_DIR, LABELS_DIR, MODEL_DIR,  # Added MODEL_DIR
    T, log_message, clear_gpu_memory, save_confusion_matrices  # Added save_confusion_matrices
)

def main():
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_message("\n==== Post-Processing Evaluation Only ====")
    
    # Set batch size for evaluation
    EVAL_BATCH_SIZE = 8  # Can be larger than training batch size for evaluation
    
    # Check for existing model checkpoint
    model_path = DISTILL_MODELS_DIR / "best_student_model.pth"
    if not model_path.exists():
        log_message(f"Error: Model checkpoint not found at {model_path}")
        return
    
    # Load the student model
    log_message(f"Loading existing model from {model_path}")
    student = MobileNetV2LSTMStudent(hidden_size=128, lstm_layers=1)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.to(DEVICE)
    student.eval()
    
    # Data transforms
    transform_student = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Prepare test dataset only
    test_ds = DAiSEERawDataset(LABELS_DIR / "TestLabels.csv", FRAMES_DIR, transform_student, 40)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create evaluator
    distiller = Distiller(None, student)  # No need for teacher ensemble during evaluation
    
    # Evaluate with original predictions
    log_message("\n=== Evaluating with original predictions ===")
    original_metrics = distiller.evaluate(test_loader, "Test", apply_postprocessing=False)
    orig_mj = METRICS_DIR / f"student_original_metrics_{timestamp}.json"
    with open(orig_mj, "w") as f:
        json.dump(original_metrics, f, indent=2)
    log_message(f"Saved original metrics => {orig_mj}")
    
    # Initialize and load ensemble
    log_message("Loading ensemble models...")
    ensemble = DAiSEEEnsemble()
    ensemble.load_all()
    
    # Evaluate with post-processing
    log_message("\n=== Evaluating with post-processing ===")
    post_metrics = distiller.evaluate(test_loader, "Test", apply_postprocessing=True)
    post_mj = METRICS_DIR / f"student_postprocessed_metrics_{timestamp}.json"
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
    
    # Clean up
    clear_gpu_memory()
    log_message("==== Post-Processing Evaluation Complete ====")
        
    # Evaluate ensemble (before post-processing)
    log_message("\n=== Evaluating ensemble (original predictions) ===")
    ensemble_metrics = ensemble.evaluate(test_loader, "Ensemble", apply_postprocessing=False)
    ensemble_mj = METRICS_DIR / f"ensemble_original_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(ensemble_mj, "w") as f:
        json.dump(ensemble_metrics, f, indent=2)
    log_message(f"Saved ensemble metrics => {ensemble_mj}")
    save_confusion_matrices(ensemble_metrics, "ensemble_original")

    # Evaluate ensemble (with post-processing)
    log_message("\n=== Evaluating ensemble (with post-processing) ===")
    ensemble_post_metrics = ensemble.evaluate(test_loader, "Ensemble", apply_postprocessing=True)
    ensemble_post_mj = METRICS_DIR / f"ensemble_postprocessed_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(ensemble_post_mj, "w") as f:
        json.dump(ensemble_post_metrics, f, indent=2)
    log_message(f"Saved post-processed ensemble metrics => {ensemble_post_mj}")

    # Generate comparison visualizations
    distiller.save_comparison_visualizations(ensemble_post_metrics, "ensemble")

    # Print summary comparing ensemble and student
    log_message("\n=== Performance Comparison Summary ===")
    for emo in EMOTIONS:
        ens_acc = ensemble_metrics[emo]["accuracy"]
        ens_post_acc = ensemble_post_metrics[emo]["post_accuracy"]
        stu_acc = original_metrics[emo]["accuracy"]
        stu_post_acc = post_metrics[emo]["post_accuracy"]
        
        log_message(f"{emo}: Ensemble {ens_acc:.4f}/{ens_post_acc:.4f} vs Student {stu_acc:.4f}/{stu_post_acc:.4f}")
    

if __name__ == "__main__":
    main()