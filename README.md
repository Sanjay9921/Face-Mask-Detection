# Face Mask Detection with PyTorch: A Comparative Study of Manual and Pretrained ResNet34

## Project Overview

This repository presents a research-focused, end-to-end deep learning pipeline for object detection, using face mask classification as a case study. It aims to deepen understanding of:
- Manual implementation versus transfer learning approaches with ResNet34
- Effects of data characteristics (class imbalance, bounding box dimensions) on model performance
- Statistics-driven EDA, model evaluation, and interpretability techniques in PyTorch

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset and EDA](#dataset-and-eda)
3. [Model Architectures](#model-architectures)
   - Manual ResNet34
   - Pretrained ResNet34
4. [Training Protocol](#training-protocol)
5. [Evaluation and Metrics](#evaluation-and-metrics)
6. [Visualization and Interpretability](#visualization-and-interpretability)
7. [How to Run](#how-to-run)
8. [References & Credits](#references--credits)

---

## Introduction

This project investigates the learning dynamics, generalization, and statistical properties of both manually built and pretrained deep convolutional models on an image dataset annotated for face mask detection. Key research questions include:
- How do hand-built and pretrained models compare in feature extraction and convergence?
- What dataset properties most strongly influence detection accuracy?
- Are advanced interpretability/visualization methods needed to understand misclassifications and model decisions?

---

## Dataset and EDA

- Face mask dataset composed of paired image (.png) and annotation (.xml) files.
- Scientific EDA includes:
  - Dataset audit (file integrity, image-annotation parity)
  - Quantitative statistics of image dimensions, aspect ratios, class distributions, and bounding box coverage
  - Advanced visualization: violin plots, bounding box spatial heatmaps, class imbalance metrics

---

## Model Architectures

- **Manual ResNet34:** Complete from-scratch implementation for didactic purposes. Includes residual block definitions, custom initialization, and in-line explanations.
- **Pretrained ResNet34:** Utilizes `torchvision.models.resnet34` pretrained on ImageNet and fine-tuned for the face mask dataset.
- Both models evaluated with identical preprocessing, losses, and optimization schemes for a rigorous side-by-side benchmark.

---

## Training Protocol

- Stratified train/val/test splits for robust evaluation
- Standard data augmentation, normalization, and reproducible seed management
- Early stopping, checkpoint management, and live statistics collection

---

## Evaluation and Metrics

- Tracking of training and validation loss curves with statistical diagnostics (mean, variance, moving average)
- Accuracy, precision, recall, F1-score, and confusion matrices reported per epoch and on final test split
- Advanced: Per-class ROC/AUC, gradient flow visualization, and model explainability

---

## Visualization and Interpretability

- Grad-CAM and feature map visualization for model interpretability
- Error analysis: Identify hard samples, visualize misclassified/badly localized cases
- Comparison of feature representations using PCA/t-SNE plots

---

## How to Run

1. Install requirements (Kaggle/Colab: pip install as outlined in the notebook).
2. Download and organize the dataset as specified in the notebook.
3. Run each notebook section sequentially, starting from EDA to model training and evaluation.
4. Results, metrics, and plots are saved into `/results/` and `/plots/` folders.

---

## References & Credits

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. [CVPR 2016]
- Dataset and annotation: SandhyaKrishnan02
- TorchVision, PyTorch community for deep learning utilities

---

## License

For academic and non-commercial use. See LICENSE file (if applicable).
