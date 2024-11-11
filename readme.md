# Enhanced Noise Reduction Project

This project implements an enhanced noise reduction model using PyTorch, focusing on real-time noise reduction for audio files.

## Features

- **Noise Reduction Model**: Uses a deep learning model with STFT, MFCC features, and enhanced spectral reconstruction.
- **Training and Evaluation**: Supports training on noisy and clean audio datasets with custom loss functions to preserve spectral and temporal fidelity.

## Techniques Implemented

- **Spectral Gating**: A noise reduction technique used for audio filtering
- **Noise Reduction Model**: A custom deep learning model, **NoiseReducer**, tailored for denoising tasks
- **Evaluation Metrics**: Signal-to-Noise Ratio (SNR) and Peak Signal-to-Noise Ratio (PSNR) for model performance measurement

## Technologies Used

- **Python**: Core programming language for model implementation and training.
- **PyTorch**: Framework for building, training, and evaluating the neural network.
- **Numpy**: Used for numerical computations.
- **DataParallel**: Multi-core CPU setup to accelerate training.
- **STFT and MFCC**: For audio feature extraction and spectral accuracy.
- **AdamW Optimizer**: Used with a OneCycleLR scheduler for optimized training.

## Directory Structure

data/ ├── clean/ └── noisy/ results/ models/

## Getting Started

1. **Install Dependencies**: Ensure Python and PyTorch are installed.
2. **Run Training**: Execute `main.py` to start training.

For more details, explore the `main.py`, `model.py`, and `trainer.py` files.
