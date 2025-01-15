# Wheat Images Classification Project

## Overview
An image classification model using deep learning to assist smallholder wheat farmers in diagnosing pests and diseases in wheat crops. The model is trained to identify various wheat crop conditions including Aphids (pest), Yellow Rust, Mildew, Fusarium Head Blight (diseases), and healthy wheat plants.

## Problem Statement
Smallholder wheat farmers need a tool to diagnose pests and diseases in wheat crops to prevent yield loss.

## Goal
Develop an image classification model for diagnosing wheat pests and diseases, which will be deployable on a web platform for smallholder farmers.

## Dataset
- Source: [Kaggle Wheat Plant Diseases Dataset](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases)
- Classes:
  - Pests: Aphid
  - Diseases: Yellow Rust, Mildew, Fusarium Head Blight
  - Healthy: Normal wheat crops

## Project Structure

wheat_images_classification/
 data/                     # Original dataset
│ train/               
│ valid/               
│ test/                
├── processed_data/          # Preprocessed images (224x224)
│ train/
│ valid/
│ test/
├── balanced_data/           # Balanced dataset
│ train/
│ valid/
│ test/
└── best_model.keras         # Saved model file

### Data Processing
- Image resizing to 224x224 pixels
- RGB format conversion
- Data augmentation for training:
  - Rotation range: 20
  - Width shift: 0.2
  - Height shift: 0.2
  - Horizontal flip
  - Zoom range: 0.15

### Model Architecture
- Base Model: MobileNetV2
  - Pretrained on ImageNet
  - Input shape: (160, 160, 3)
- Additional Layers:
  - Global Average Pooling
  - Dense (128 units, ReLU)
  - Dropout (0.3)
  - Output Dense (5 units, Softmax)

### Training Configuration
- Batch Size: 128
- Image Size: 160x160
- Learning Rate: 0.0004
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

### Callbacks
- Early Stopping:
  - Monitor: val_accuracy
  - Patience: 6
  - Restore best weights
- Learning Rate Reduction:
  - Monitor: val_accuracy
  - Factor: 0.5
  - Patience: 3
  - Min learning rate: 1e-6
- Model Checkpoint:
  - Monitor: val_accuracy
  - Save best only
  ### Dependencies
     tensorflow numpy pandas pillow opencv-python matplotlib seaborn scikit-learn imbalanced-learn
### Model Performance
- Training history tracked for:
  - Accuracy
  - Validation accuracy
  - Loss
  - Validation loss
- Evaluation metrics:
  - Test accuracy
  - Confusion matrix
  - Classification report per class

### Future Improvements
- Mobile application development
- Expanded dataset with more disease categories
- Real-time prediction capabilities
- Optimization for edge devices
