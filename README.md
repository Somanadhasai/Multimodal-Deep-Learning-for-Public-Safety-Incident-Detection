# Multimodal Deep Learning for Public Safety Incident Detection

## Project Overview

This project presents a Multimodal Transformer-Based Framework for Public Safety Incident Detection using surveillance video data.

Unlike traditional approaches that rely only on visual information, the proposed framework integrates RGB appearance features, Optical Flow motion features, and Audio features to improve violence detection performance.

The model is evaluated on the XD-Violence Dataset and achieves a Mean Average Precision (mAP) of 88.52%.

---

## Problem Statement

Manual monitoring of surveillance videos is time-consuming, error-prone, and difficult to scale.

The objective of this project is to automatically detect violent incidents by combining multiple modalities:

* RGB Appearance Information
* Optical Flow Motion Information
* Audio Context Information

This multimodal approach enables the model to capture richer contextual information compared to single-modality methods.

---

## Dataset

### XD-Violence Dataset

The project uses the XD-Violence Dataset, a large-scale benchmark dataset for violence and anomaly detection in surveillance videos.

### Event Categories

| Label | Event        |
| ----- | ------------ |
| A     | Normal       |
| B1    | Fighting     |
| B2    | Shooting     |
| B4    | Riot         |
| B6    | Car Accident |
| G     | Explosion    |

---

## Feature Extraction

The framework operates on pre-extracted features rather than raw videos.

### Visual Features

* RGB Features extracted using I3D
* Optical Flow Features extracted using I3D

### Audio Features

* Audio Embeddings extracted using VGGish

This reduces computational complexity while preserving important spatio-temporal information.

---

## Preprocessing

The dataset contains videos of varying lengths.

To ensure consistent input representation:

* Uniform temporal sampling is applied
* Sequence length is fixed to 300 segments
* Shorter sequences are padded with zeros

This allows efficient batch processing and stable training.

---

## Proposed Architecture

### Stage 1: Modality-Specific Encoding

Each modality is processed independently:

* RGB Transformer Encoder
* Optical Flow Transformer Encoder
* Audio Transformer Encoder

These encoders learn temporal dependencies within each modality.

### Stage 2: Feature Projection

Encoded features are projected from:

1024 → 512 dimensions

This creates a shared embedding space across all modalities.

### Stage 3: Attention-Based Multimodal Fusion

Projected RGB, Optical Flow, and Audio representations are fused using an attention mechanism.

This enables the model to learn relationships across different modalities.

### Stage 4: Self-Attention Refinement

Multi-Head Self-Attention is applied on the fused representation to enhance temporal understanding.

### Stage 5: Feature Aggregation

Average Pooling is used to convert temporal feature sequences into a fixed-length representation.

### Stage 6: Classification

The aggregated representation is passed through:

* Fully Connected Layer
* Sigmoid Activation

for Multi-Label Classification.

---

## Model Pipeline

XD-Violence Dataset

↓

RGB (I3D)

Optical Flow (I3D)

Audio (VGGish)

↓

Transformer Encoders

↓

Feature Projection (1024 → 512)

↓

Attention-Based Fusion

↓

Multi-Head Self-Attention

↓

Average Pooling

↓

Fully Connected Layer

↓

Sigmoid Activation

↓

Violence Classification

---

## Results

| Metric                       | Score  |
| ---------------------------- | ------ |
| Mean Average Precision (mAP) | 88.52% |
| Macro F1 Score               | 0.81   |

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Scikit-Learn
* Transformer Networks
* Deep Learning
* Computer Vision
* Audio Processing

---

## License

MIT License
