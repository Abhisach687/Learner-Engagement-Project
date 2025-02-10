import os
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import logging
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Helper: Convert raw video stem to CSV clip ID.
def get_csv_clip_id(video_stem: str) -> str:
    """
    Custom mapping: returns some modified ID if needed.
    Adjust or remove if your project doesn't require special ID mapping.
    """
    if video_stem.startswith("110001"):
        return "202614" + video_stem[-4:]
    else:
        return video_stem

class DAiSEEDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, seq_length=15):
        self.root = Path(root)
        self.transform = transform
        self.seq_length = seq_length
        self.video_paths = []
        self.labels = []
        
        # Read CSV and strip extra whitespace from column names.
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Fix columns like "Frustration " -> "Frustration"
        
        # Create a dictionary for quick lookup of existing folders in self.root.
        existing_folders = {d.name: d for d in self.root.iterdir() if d.is_dir()}
        
        for idx, row in df.iterrows():
            try:
                # Use the "ClipID" column for the clip identifier.
                clip_id_raw = str(row["ClipID"]).strip()
                # Remove file extension (e.g., ".avi", ".mp4") by always taking the part before the first dot.
                clean_clip_id = clip_id_raw.split('.')[0]
                # Apply the same mapping function used in 001_data.ipynb
                mapped_clip_id = get_csv_clip_id(clean_clip_id)
                
                # First try the direct mapping.
                video_dir = self.root / mapped_clip_id
                if not video_dir.exists():
                    # If direct match fails, try to locate a folder whose name contains the clean_clip_id.
                    candidates = [d for name, d in existing_folders.items() if clean_clip_id in name]
                    if candidates:
                        video_dir = candidates[0]
                    else:
                        raise ValueError(f"Video directory for clip_id {clean_clip_id} not found in {self.root}")
                
                self.video_paths.append(video_dir)
                # Use the remaining columns as the label.
                label_values = row.drop("ClipID").tolist()
                self.labels.append(label_values)
            except Exception as e:
                clip = row.get("ClipID", "unknown")
                logger.error(f"Skipping row {idx} ({clip}): {str(e)}")
                continue

        if len(self.video_paths) == 0:
            raise ValueError(f"No valid video sequences found in {csv_path}")
        
        logger.info(f"Loaded {len(self.video_paths)} valid video sequences")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Get list of frame paths, sort them to ensure chronological order.
        frame_paths = sorted(list(self.video_paths[idx].rglob("*.jpg")))
        # Take the first `seq_length` frames.
        frames = frame_paths[:self.seq_length]
        frame_tensors = []
        for path in frames:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)
        
        sequence = torch.stack(frame_tensors)  # [seq_len, C, H, W]
        return sequence, torch.tensor(self.labels[idx], dtype=torch.float32)

# Register the dataset class for safe deserialization.
torch.serialization.add_safe_globals({"DAiSEEDataset": DAiSEEDataset})

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_LSTM, self).__init__()
        # Feature extractor using ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Updated line
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # LSTM to capture temporal features
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        
        # Fully connected layer for regression
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Take the last output
        out = self.fc(lstm_out)
        return out

def get_dataloaders(batch_size=16):
    base_path = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project/data/DAiSEE")
    # Use the proper subfolder for each split.
    train_frames_root = base_path / "ExtractedFrames" / "Train"
    val_frames_root   = base_path / "ExtractedFrames" / "Validation"
    test_frames_root  = base_path / "ExtractedFrames" / "Test"
    
    # The labels are stored under this folder
    labels_path = base_path / "DataSet/Labels"
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with the proper root folder for each split.
    train_dataset = DAiSEEDataset(root=train_frames_root, csv_path=labels_path / "TrainLabels.csv", transform=train_transform)
    val_dataset   = DAiSEEDataset(root=val_frames_root,   csv_path=labels_path / "ValidationLabels.csv", transform=test_transform)
    test_dataset  = DAiSEEDataset(root=test_frames_root,  csv_path=labels_path / "TestLabels.csv", transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return np.mean((all_preds - all_labels)**2, axis=0), r2_score(all_labels, all_preds)

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = np.inf
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return train_losses, val_losses

if __name__ == "__main__":
    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # Initialize model
    model = CNN_LSTM(num_classes=4)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_mse, test_r2 = evaluate(model, test_loader)
    logger.info(f"Test MSE: {test_mse}")
    logger.info(f"Test R2 Score: {test_r2}")