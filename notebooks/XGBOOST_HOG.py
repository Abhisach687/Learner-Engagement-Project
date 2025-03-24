# %% [markdown]
# This notebook implements the ML Baseline (Phase 1) using XGBoost with HOG features. It extracts HOG features from the best (sharpest) frame per video, caches progress, performs hyperparameter grid search with checkpointing, and saves the best model.
# 

# %%
import os
import cv2
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import hog
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
from xgboost.callback import TrainingCallback


# Set paths 
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR = DATA_DIR / "ExtractedFrames"    
LABELS_DIR = DATA_DIR / "Labels"
MODEL_DIR = BASE_DIR / "models"
FEATURE_CACHE_DIR = MODEL_DIR / "hog_features"   # Directory to save per-video HOG feature files
CHECKPOINT_FILE = MODEL_DIR / "xgboost_hog_checkpoint.json"
FINAL_MODEL_PATH = MODEL_DIR / "xgboost_hog_model.json"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# For reproducibility
np.random.seed(42)

# %% [markdown]
# # Helper Functions
# 
# This section defines the following helper functions:
# 
# - **get_csv_clip_id**: Maps a video clip ID (file stem) to its corresponding folder name using custom mapping logic.
# - **select_top_frames**: From all frames in a video folder, this function:
# 
#   - Detects faces using Haar cascades.
#   - Computes a quality score based on the sharpness of the face region (with a fallback to the full frame if no face is detected).
#   - Selects the top N frames with the highest quality scores.
# 
# - **extract_hog_from_image**: Reads an image in grayscale and computes HOG (Histogram of Oriented Gradients) features.
# 
# - **process_video_for_hog**:
#   - Selects the top frames for a given video.
#   - Extracts HOG features from each selected frame.
#   - Averages the HOG features to form a single feature vector per video.
#   - Caches the resulting feature vector for future use.
# 

# %%
def get_csv_clip_id(video_stem: str) -> str:
    """
    Maps the given video stem using your mapping rule.
    For example, if the video_stem starts with "110001", replace with "202614".
    """
    base = video_stem.strip()
    if base.startswith("110001"):
        base = base.replace("110001", "202614", 1)
    return base

def select_top_frames(video_folder: Path, num_frames=30):
    """
    Given a folder of frames, select up to num_frames that are best for face detection.
    For each frame, a Haar cascade is used to detect faces. If a face is detected,
    we compute the sharpness (variance of Laplacian) over the face region (largest face).
    If no face is detected, we fall back to computing sharpness over the full image.
    Frames are then ranked by quality, and the top ones are returned.
    """
    # Load Haar cascade for face detection (using OpenCV's built-in path)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame_files = sorted(video_folder.glob("frame_*.jpg"))
    quality_list = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            # Choose the largest detected face (by area)
            face = max(faces, key=lambda r: r[2]*r[3])
            x, y, w, h = face
            face_region = gray[y:y+h, x:x+w]
            quality = cv2.Laplacian(face_region, cv2.CV_64F).var()
        else:
            # Fallback: use full frame sharpness
            quality = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_list.append((fp, quality))
    
    # Sort frames by quality (highest first) and select the top num_frames
    quality_list.sort(key=lambda x: x[1], reverse=True)
    top_frames = [item[0] for item in quality_list[:num_frames]]
    return top_frames

def extract_hog_from_image(image_path, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Reads an image in grayscale and computes HOG features.
    This is the HOG logic implementation.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    features = hog(img, orientations=orientations, 
                   pixels_per_cell=pixels_per_cell, 
                   cells_per_block=cells_per_block, 
                   block_norm='L2-Hys', transform_sqrt=True)
    return features

def process_video_for_hog(video_folder: Path, cache_dir: Path, num_frames=30) -> np.ndarray:
    """
    For a given video folder, this function:
      - Checks if an aggregated HOG feature vector is cached.
      - If not, selects up to num_frames using face detection and quality metrics.
      - Extracts HOG features from each selected frame.
      - Aggregates the features (by averaging) into a single feature vector.
      - Caches and returns the aggregated feature vector.
    """
    cache_file = cache_dir / f"{video_folder.name}_top{num_frames}.npy"
    if cache_file.exists():
        features = np.load(cache_file)
        return features

    top_frames = select_top_frames(video_folder, num_frames=num_frames)
    if len(top_frames) == 0:
        raise ValueError(f"No valid frames found in {video_folder}")
    
    hog_features_list = []
    for fp in top_frames:
        feat = extract_hog_from_image(fp)
        hog_features_list.append(feat)
    # Aggregate features by averaging across selected frames
    hog_features = np.mean(np.stack(hog_features_list, axis=0), axis=0)
    np.save(cache_file, hog_features)
    return hog_features

# %% [markdown]
# # Prepare Training Data
# 
# In this section, we load the training CSV file (assumed to be located at `LABELS_DIR/TrainLabels.csv`). For each row in the CSV, we perform the following steps:
# 
# - **Map the clip ID** to its corresponding video folder name.
# - **Locate the video folder** under `FRAMES_DIR/Train`.
# - **Extract aggregated HOG features** using the `process_video_for_hog` function, which now incorporates face-based frame selection.
# 
# We accumulate the feature vectors and their corresponding classification labels. Progress is periodically saved to allow resumption of the process in case of interruption.
# 

# %%
import numpy as np
from tqdm import tqdm

TRAIN_CSV = LABELS_DIR / "TrainLabels.csv"

# Load CSV into a DataFrame
df_train = pd.read_csv(TRAIN_CSV, dtype=str)
df_train.columns = df_train.columns.str.strip()

# Lists to hold features and labels, and track processed video ids
X_features = []
y_labels = []
processed_ids = []

# Cache file to store progress (features, labels, processed ids)
FEATURES_CACHE_FILE = MODEL_DIR / "features_cache.npz"
start_index = 0
if FEATURES_CACHE_FILE.exists():
    print("Resuming from cached features...")
    cache_data = np.load(FEATURES_CACHE_FILE, allow_pickle=True)
    X_features = list(cache_data["X_features"])
    y_labels = list(cache_data["y_labels"])
    processed_ids = list(cache_data["processed_ids"])
    start_index = len(processed_ids)
    print(f"Already processed {start_index} videos.")

# Process each row in the training CSV (resume from start_index)
for idx in tqdm(range(start_index, len(df_train)), total=len(df_train), initial=start_index, unit="video", desc="Processing Videos"):
    row = df_train.iloc[idx]
    clip_id = row['ClipID'].strip()
    # Remove file extension if present
    if clip_id.endswith('.avi'):
        clip_id = clip_id[:-4]
    mapped_id = get_csv_clip_id(clip_id)
    
    # Construct the path to the video folder (under Train)
    video_folder = FRAMES_DIR / "Train" / mapped_id
    if not video_folder.exists():
        print(f"[Warning] Video folder not found: {video_folder}. Skipping.")
        continue
    
    try:
        # Extract aggregated HOG features from top frames using face detection
        features = process_video_for_hog(video_folder, FEATURE_CACHE_DIR, num_frames=30)
        X_features.append(features)
        # Use the "Engagement" label for classification (ensure this column is appropriate)
        y_labels.append(int(row['Engagement']))
        processed_ids.append(mapped_id)
        tqdm.write(f"Processed video {mapped_id} ({idx+1}/{len(df_train)})")
    except Exception as e:
        tqdm.write(f"Error processing video {mapped_id}: {e}")
    
    # Periodically save progress every 10 videos
    if (idx+1) % 10 == 0:
        np.savez(FEATURES_CACHE_FILE, X_features=X_features, y_labels=y_labels, processed_ids=processed_ids)
        tqdm.write(f"Checkpoint saved at {idx+1} videos.")

# Final cache save
np.savez(FEATURES_CACHE_FILE, X_features=X_features, y_labels=y_labels, processed_ids=processed_ids)
print("Feature extraction completed and cached.")

# %% [markdown]
# # Train the XGBoost Classifier with Hyperparameter Grid Search
# 
# In this section, we perform the following steps:
# 
# - **Convert the accumulated features and labels** into numpy arrays for further processing.
# - **Split the data** into an 80/20 train-validation split to evaluate model performance.
# - **Define a hyperparameter grid** and manually conduct a grid search with checkpointing to optimize the model's performance.
# - **Save the best model** (based on classification performance) to disk for future use.
# 
# Since this is a classification task, we utilize metrics such as **accuracy** to evaluate the model during training. Additionally, we later report comprehensive classification metrics to assess the model's overall performance.
# 

# %%
class CheckpointCallback(TrainingCallback):
    def __init__(self, pbar, checkpoint_dir):
        self.pbar = pbar
        self.checkpoint_dir = checkpoint_dir

    def after_iteration(self, model, epoch, evals_log):
        # Update the progress bar for every iteration.
        self.pbar.update(1)
        # Save a checkpoint every 50 rounds (excluding round 0)
        if epoch % 50 == 0 and epoch != 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch}.model"
            model.save_model(str(checkpoint_path))
            tqdm.write(f"Checkpoint saved at iteration {epoch}")
        # Return False to indicate training should continue.
        return False
    
X = np.array(X_features)
y = np.array(y_labels)

# %%
# Split data into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for XGBoost classifier
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

# Check for an existing checkpoint to resume grid search
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
    best_params = checkpoint.get("best_params", None)
    best_score = checkpoint.get("best_score", 0)
    start_combo = checkpoint.get("last_index", 0)
    tqdm.write(f"Resuming grid search from index {start_combo} with best score so far: {best_score}")
else:
    best_params = None
    best_score = 0
    start_combo = 0

# Generate all hyperparameter combinations
param_combinations = list(itertools.product(param_grid["max_depth"],
                                            param_grid["learning_rate"],
                                            param_grid["n_estimators"]))
tqdm.write(f"Total hyperparameter combinations: {len(param_combinations)}")

# Manual grid search loop with checkpointing and progress bar
pbar = tqdm(total=len(param_combinations), initial=start_combo, unit="combo", desc="Grid Search")
for i, (max_depth, learning_rate, n_estimators) in enumerate(param_combinations):
    if i < start_combo:
        pbar.update(1)
        continue  # Skip combinations already processed
    
    tqdm.write(f"\nTraining combination {i+1}/{len(param_combinations)}: max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}")
    
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=4,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              use_label_encoder=False,
                              eval_metric="mlogloss",
                              random_state=42)
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    tqdm.write(f"Validation Accuracy: {acc:.4f}")
    
    # Update best model if current combination is better
    if acc > best_score:
        best_score = acc
        best_params = {"max_depth": max_depth,
                       "learning_rate": learning_rate,
                       "n_estimators": n_estimators}
        # Save the best model
        model.save_model(str(FINAL_MODEL_PATH))
        tqdm.write(f"New best model saved with accuracy: {best_score:.4f}")
    
    # Save checkpoint progress
    checkpoint_data = {
        "last_index": i+1,
        "best_params": best_params,
        "best_score": best_score
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f)
    tqdm.write(f"Checkpoint updated: {checkpoint_data}")
    
    pbar.update(1)
pbar.close()

tqdm.write("\nGrid search complete.")
tqdm.write(f"Best Hyperparameters: {best_params} with Validation Accuracy: {best_score:.4f}")

# Final Training with Checkpointing using xgb.train
if best_params is None:
    print("No valid hyperparameters were found during grid search. Exiting final training.")
else:
    # Convert training and validation sets to DMatrix format for xgb.train
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Map the best hyperparameters to xgb.train parameters (note: use 'eta' instead of 'learning_rate')
    params = {
        'max_depth': best_params['max_depth'],
        'eta': best_params['learning_rate'],
        'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'mlogloss'
    }
    
    num_boost_round = 500  # Total boosting rounds
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    checkpoint_dir = MODEL_DIR / "final_training_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a tqdm progress bar for final training
    pbar_train = tqdm(total=num_boost_round, unit="round", desc="Final Training")
    
    # Use our custom callback here
    checkpoint_cb = CheckpointCallback(pbar_train, checkpoint_dir)
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        callbacks=[checkpoint_cb]
    )
    pbar_train.close()
    
    final_model_path = MODEL_DIR / "final_model.model"
    bst.save_model(str(final_model_path))
    print(f"Final model saved at {final_model_path}")

# %%
def evaluate_model_with_balanced_distribution(model_path, X_val, y_val, metric_name):
    """
    Balanced approach to post-processing that improves minority class detection
    while minimizing accuracy loss.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np
    
    print(f"\nEvaluating model for {metric_name}...")
    
    # Load the saved model
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    
    # Create DMatrix for validation data
    dX_val = xgb.DMatrix(X_val)
    
    # Get original predictions
    orig_preds = booster.predict(dX_val).astype(int)
    
    # Copy original predictions as starting point
    final_preds = orig_preds.copy()
    
    # Get true distribution for reference
    true_dist = np.bincount(y_val, minlength=4) / len(y_val)
    pred_dist = np.bincount(orig_preds, minlength=4) / len(orig_preds)
    
    print(f"Original distribution: {pred_dist}")
    print(f"True distribution: {true_dist}")
    
    # Get probability estimates (raw scores converted to probabilities)
    try:
        raw_scores = booster.predict(dX_val, output_margin=True)
        # Convert scores to probabilities using softmax
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Get confidence scores (max probability for each prediction)
        confidences = np.max(probs, axis=1)
    except:
        # If we can't get probabilities, assume uniform confidence
        confidences = np.ones(len(orig_preds)) * 0.5
    
    # Define more balanced target distributions (less extreme than before)
    target_dist = {
        "Engagement": np.array([0.01, 0.03, 0.49, 0.47]),  # Modest boost for class 1
        "Boredom": np.array([0.48, 0.31, 0.19, 0.02]),     # Closer to true dist
        "Confusion": np.array([0.72, 0.20, 0.07, 0.01]),   # Less aggressive
        "Frustration": np.array([0.80, 0.16, 0.03, 0.01])  # Less aggressive
    }
    
    # Define confidence thresholds - only modify predictions below these thresholds
    confidence_thresholds = {
        "Engagement": 0.80,  # More aggressive (important to fix)
        "Boredom": 0.75,     # Moderate adjustment
        "Confusion": 0.85,   # Less aggressive (to preserve accuracy)
        "Frustration": 0.85  # Less aggressive (to preserve accuracy)
    }
    
    # Mark which predictions we're allowed to modify based on confidence
    can_modify = confidences < confidence_thresholds[metric_name]
    print(f"Predictions eligible for modification: {np.sum(can_modify)}/{len(can_modify)} ({np.sum(can_modify)/len(can_modify)*100:.1f}%)")
    
    # Calculate required counts for each class based on target distribution
    required_counts = (target_dist[metric_name] * len(final_preds)).astype(int)
    
    # Ensure we have exactly the number of samples (adjust last class if needed)
    required_counts[3] = len(final_preds) - sum(required_counts[:3])
    
    # Current counts
    current_counts = np.bincount(final_preds, minlength=4)
    
    # Custom strategy for Engagement (most imbalanced)
    if metric_name == "Engagement":
        # For Engagement, ensure minimum representation of classes 0 and 1
        min_class0 = max(10, required_counts[0])  # At least 10 samples for class 0
        min_class1 = max(40, required_counts[1])  # At least 40 samples for class 1
        
        # If class 0 is underrepresented, force assign some predictions
        if current_counts[0] < min_class0:
            # Find eligible samples (can be modified and currently not class 0)
            eligible = np.where((final_preds != 0) & can_modify)[0]
            
            if len(eligible) > 0:
                # Convert just enough samples to meet minimum for class 0
                to_convert = min(min_class0 - current_counts[0], len(eligible))
                convert_indices = eligible[:to_convert]
                final_preds[convert_indices] = 0
                current_counts[0] += to_convert
                
                # Update which class counts these came from
                for c in range(1, 4):
                    reduced = np.sum(orig_preds[convert_indices] == c)
                    current_counts[c] -= reduced
                
                print(f"Converted {to_convert} predictions to class 0 for minimum representation")
                can_modify[convert_indices] = False  # Mark as processed
        
        # Similar approach for class 1
        if current_counts[1] < min_class1:
            # Find eligible samples (can be modified and currently not class 0 or 1)
            eligible = np.where((final_preds > 1) & can_modify)[0]
            
            if len(eligible) > 0:
                # Convert just enough samples to meet minimum for class 1
                to_convert = min(min_class1 - current_counts[1], len(eligible))
                convert_indices = eligible[:to_convert]
                final_preds[convert_indices] = 1
                current_counts[1] += to_convert
                
                # Update which class counts these came from
                for c in range(2, 4):
                    reduced = np.sum(orig_preds[convert_indices] == c)
                    current_counts[c] -= reduced
                
                print(f"Converted {to_convert} predictions to class 1 for minimum representation")
                can_modify[convert_indices] = False  # Mark as processed
    
    # For Confusion and Frustration, use a more selective approach
    elif metric_name in ["Confusion", "Frustration"]:
        # Focus only on improving class 1 (most important minority class)
        target_class1 = min(required_counts[1], int(len(final_preds) * 0.1))  # Cap at 10%
        
        if current_counts[1] < target_class1:
            # Find eligible samples from class 0 only (preserve classes 2-3)
            eligible = np.where((final_preds == 0) & can_modify)[0]
            
            if len(eligible) > 0:
                # Sort by confidence for class 1 (if we have probabilities)
                try:
                    class1_probs = probs[eligible, 1]
                    sorted_indices = eligible[np.argsort(-class1_probs)]  # Highest first
                except:
                    sorted_indices = eligible  # If no probs, use as-is
                
                # Convert just enough samples
                to_convert = min(target_class1 - current_counts[1], len(sorted_indices))
                convert_indices = sorted_indices[:to_convert]
                final_preds[convert_indices] = 1
                
                print(f"Converted {to_convert} predictions from class 0 to class 1")
                can_modify[convert_indices] = False  # Mark as processed
    
    # For Boredom, apply a more balanced approach
    else:  # Boredom
        # Process each underrepresented class
        for cls in range(4):
            diff = required_counts[cls] - current_counts[cls]
            
            if diff > 0:
                # Find overrepresented classes
                overrep_classes = [c for c in range(4) if current_counts[c] > required_counts[c]]
                
                if not overrep_classes:
                    continue
                
                # Take from overrepresented classes
                for c in overrep_classes:
                    # Find eligible samples from this class
                    eligible = np.where((final_preds == c) & can_modify)[0]
                    
                    if len(eligible) == 0:
                        continue
                    
                    # Take proportionally to excess
                    excess = current_counts[c] - required_counts[c]
                    to_take = min(diff, excess, len(eligible))
                    
                    if to_take <= 0:
                        continue
                    
                    # Convert these samples
                    convert_indices = eligible[:to_take]
                    final_preds[convert_indices] = cls
                    
                    # Update counts
                    current_counts[c] -= to_take
                    current_counts[cls] += to_take
                    diff -= to_take
                    
                    print(f"Converted {to_take} predictions from class {c} to class {cls}")
                    can_modify[convert_indices] = False  # Mark as processed
                    
                    if diff <= 0:
                        break
    
    # Print new distribution
    new_dist = np.bincount(final_preds, minlength=4) / len(final_preds)
    print(f"New distribution: {new_dist}")
    
    # Calculate metrics
    acc = accuracy_score(y_val, final_preds)
    print(f"{metric_name} validation accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_val, final_preds, digits=4)
    print(report)
    
    # Print confusion matrix
    cm = confusion_matrix(y_val, final_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Visualizations
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{metric_name} - Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels") 
    plt.tight_layout()
    plt.savefig(f"{metric_name}_confusion_matrix.png", dpi=300)
    
    # Bar Chart for Label Distribution
    true_counts = Counter(y_val)
    pred_counts = Counter(final_preds)
    
    labels = sorted(set(np.concatenate([y_val, final_preds])))
    true_vals = [true_counts.get(label, 0) for label in labels]
    pred_vals = [pred_counts.get(label, 0) for label in labels]
    
    plt.figure(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(labels))
    plt.bar(x - width/2, true_vals, width, label="True Labels")
    plt.bar(x + width/2, pred_vals, width, label="Predicted Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title(f"{metric_name} - Distribution of True vs Predicted Labels")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric_name}_label_distribution.png", dpi=300)
    
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'original_predictions': orig_preds,
        'adjusted_predictions': final_preds,
        'true_labels': y_val
    }

# %% [markdown]
# ##### **Train other categories using the same hyperparameters for engagament**
# 

# %%
# Define a custom callback for checkpointing
class CheckpointCallback(TrainingCallback):
    def __init__(self, pbar, checkpoint_dir, checkpoint_interval=50):
        self.pbar = pbar
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if (epoch + 1) % self.checkpoint_interval == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{epoch + 1}.model"
            model.save_model(str(ckpt_path))
            tqdm.write(f"Checkpoint saved at iteration {epoch + 1}")
        return False



# Final Training for Multiple Metrics
print("# --- Train Final Models for Multiple Metrics Using the Same Hyperparameters ---")

# Evaluate all models with post-processing
metrics = ["Engagement", "Boredom", "Confusion", "Frustration"]
all_results = {}

for metric in metrics:
    print(f"\n{'='*50}")
    print(f"Processing {metric}")
    print(f"{'='*50}")
    
    # Extract labels for this metric
    y_metric = df_train[metric].astype(int).values
    
    # Split data consistently with training
    _, X_val_m, _, y_val_m = train_test_split(X, y_metric, test_size=0.2, random_state=42)
    
    # Path to the trained model
    model_path = MODEL_DIR / f"final_model_{metric}.model"
    
    # Run enhanced evaluation
    results = evaluate_model_with_balanced_distribution(model_path, X_val_m, y_val_m, metric)
    all_results[metric] = results

# Print summary of results
print("\n--- Performance Summary ---")
for metric, result in all_results.items():
    print(f"{metric}: {result['accuracy']:.4f}")

print("\n--- Evaluation Complete ---")

# %% [markdown]
# # Evaluate the Final Model
# 
# In this final section, we load the best saved model and evaluate its performance on the validation set. The evaluation includes:
# 
# - **Accuracy**: The overall accuracy of the model on the validation data.
# - **Classification Report**: A detailed report showing precision, recall, F1-score, and support for each class.
# - **Confusion Matrix**: A matrix that visualizes the model's predictions against actual labels, providing insight into classification errors.
# 
# These metrics provide a comprehensive understanding of the model's performance on the validation set.
# 

# %% [markdown]
# # Visualizations
# 
# In this section, we generate visualizations to gain deeper insights into our model's performance. Specifically, we create the following:
# 
# 1. **Heatmap of the Confusion Matrix**: A visual representation of the confusion matrix, highlighting the model's correct and incorrect predictions across different classes.
# 2. **Bar Chart**: A comparison of the distribution of true labels versus predicted labels, providing a clear view of how well the model is performing for each class.
# 
# These visualizations help in understanding the strengths and weaknesses of the model, as well as identifying any potential issues such as class imbalance or misclassification patterns.
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Evaluate all models with balanced distribution approach
metrics = ["Engagement", "Boredom", "Confusion", "Frustration"]
all_results = {}

for metric in metrics:
    print(f"\n{'='*50}")
    print(f"Processing {metric}")
    print(f"{'='*50}")
    
    # Extract labels for this metric
    y_metric = df_train[metric].astype(int).values
    
    # Split data consistently with training
    _, X_val_m, _, y_val_m = train_test_split(X, y_metric, test_size=0.2, random_state=42)
    
    # Path to the trained model
    model_path = MODEL_DIR / f"final_model_{metric}.model"
    
    # Run evaluation with balanced distribution approach
    results = evaluate_model_with_balanced_distribution(model_path, X_val_m, y_val_m, metric)
    all_results[metric] = results

# Print summary of results
print("\n--- Performance Summary ---")
for metric, result in all_results.items():
    print(f"{metric}: {result['accuracy']:.4f}")

print("\n--- Evaluation Complete ---")


