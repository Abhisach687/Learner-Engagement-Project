This project focuses on detecting and analyzing learner engagement states using machine learning and deep learning models. It identifies four key emotional states: Engagement, Boredom, Confusion, and Frustration.

## Project Overview

The system uses a combination of machine learning and deep learning models to detect learner engagement in real time. It implements various fusion strategies to optimize both accuracy and latency, making it suitable for real-time applications.

## Note:

- The files use absolute paths for model loading and data processing. Ensure to adjust these paths according to your local setup.

- Some data sanitization might not work as DAiSEE label csv files seem to have a trailing space in the label column. This can be fixed by manually editing the CSV files or adjusting the code to strip whitespace due to limited timeline for the project. Adjust accordingly if you encounter issues.

- The code was written and tested on Windows 11 with Python 3.12.1. Some dependencies may require specific versions to work correctly. A virtual environment is recommended to manage dependencies. Or, adjust the code to work with your environment.

- Adjust hyperparameters and model configurations based on your hardware capabilities, operating system and performance requirements.

## Key Features

- **Multi-emotion detection**: Analyzes Engagement, Boredom, Confusion, and Frustration
- **Real-time processing**: Optimized for webcam-based detection
- **Multiple fusion strategies**: Includes various model fusion techniques for optimal performance
- **Performance visualization**: Generates confusion matrices and performance metrics
- **Benchmarking tools**: Compares different fusion strategies for latency and accuracy

## Model Architectures

The project explores several neural network architectures:

- **EfficientNetV2L + BiLSTM**: Combines spatial features with bidirectional LSTM for temporal reasoning
- **EfficientNetV2L + TCN**: Uses Temporal Convolutional Networks for sequence modeling
- **Cross-Attention mechanisms**: Enhances feature interaction between spatial and temporal components
- **CBAM (Convolutional Block Attention Module)**: Refines attention in both channel and spatial dimensions

## Fusion Strategies

The system implements several fusion strategies that can be benchmarked:

- `all_mobilenet`: Uses MobileNet predictions exclusively
- `all_xgboost`: Uses XGBoost predictions exclusively
- `gated_fusion`: Uses confidence-based gating between models
- `selective_fusion`: Selects the best model per emotion
- `weighted_fusion`: Weighted combination of model outputs
- `hybrid_balanced_fusion`: Hybrid approach balancing latency and accuracy
- `mobilenet_confidence_gate`: Confidence-based gating system

## Usage

### Running the Application

```bash
python app.py
```

### Benchmarking Fusion Strategies

Run the real-time latency analysis tool:

```bash
python realtime_latency.py
```

While the tool is running:

- Press 'b' to toggle auto-benchmark mode
- Press '1-7' to manually switch between fusion strategies
- Press 'q' to quit

## Results

The project includes pre-generated visualizations:

- Confusion matrices for each emotion
- F1 score distributions
- Label distributions

## Dataset

The project uses the DAiSEE dataset (not included in the repository). The data preprocessing pipeline is available in 001_DataExtract.ipynb.

## Project Structure

- app.py: Main application entry point
- realtime_latency.py: Tool for measuring and analyzing latency of different fusion strategies
- export_onnx.py: Utility for exporting models to ONNX format
- notebooks: Jupyter notebooks for model development and experimentation
- models: Directory for storing trained models (not tracked in git)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- XGBoost
- NumPy
- Matplotlib
- Seaborn etc.

## License

This project is intended for educational and research purposes. The DAiSEE dataset is not included in this repository due to licensing restrictions. Users must obtain the dataset separately for training and evaluation.

This project is licensed under the terms of the MIT license.
