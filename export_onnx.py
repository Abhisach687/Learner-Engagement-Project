import torch
import os
from pathlib import Path
import sys

# Add the current directory to the path so we can import from app.py
sys.path.append('.')

# Import the model definition
from app import MobileNetV2LSTMStudent, MODEL_DIR, DISTILL_MODEL_PATH, FRAME_HISTORY

def export_model_to_onnx():
    print("Starting ONNX export process...")
    
    # Check if paths exist
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)
    
    if not DISTILL_MODEL_PATH.exists():
        print(f"Error: Model file not found at {DISTILL_MODEL_PATH}")
        return False
    
    onnx_path = MODEL_DIR / "student_model.onnx"
    
    # Create the model
    model = MobileNetV2LSTMStudent(hidden_size=128, lstm_layers=1)
    
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
    
    # Create dummy input with batch_size=1 to avoid the warning
    print(f"Creating dummy input with shape: [1, {FRAME_HISTORY}, 3, 224, 224]")
    dummy_input = torch.randn(1, FRAME_HISTORY, 3, 224, 224, device='cpu')
    
    print(f"Exporting model to {onnx_path}")
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
    
    print(f"ONNX export complete! Model saved to {onnx_path}")
    return True

if __name__ == "__main__":
    success = export_model_to_onnx()
    if success:
        print("Success! You can now run app.py to use the ONNX model.")
    else:
        print("Export failed. Please check the error messages.")