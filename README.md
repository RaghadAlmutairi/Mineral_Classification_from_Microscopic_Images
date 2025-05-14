# Mineral Microscopic Image Classification with MobileNetV2 and Custom CNNs

This repository contains code and experiments for classifying minerals from microscopic images using deep learning. The project compares two custom Convolutional Neural Networks (CNNs) and a transfer learning approach based on MobileNetV2.

## Dataset

- **Source:** [Mineral Microscopic Image Dataset on Kaggle](https://www.kaggle.com/datasets/jlexzhong/mineral-microscopic-image-dataset)
- **Description:** 671 microscopic images labeled into 8 mineral classes. Original labels are in Chinese and translated to English in this project.

## Features

- Data loading and preprocessing, including label translation.
- Training/validation split with stratification.
- Three model architectures:
  - Custom CNN 1 (shallow)
  - Custom CNN 2 (deeper)
  - MobileNetV2 (transfer learning)
- Data augmentation for robust training.
- Evaluation: accuracy, loss, classification report, and confusion matrix for each model.



