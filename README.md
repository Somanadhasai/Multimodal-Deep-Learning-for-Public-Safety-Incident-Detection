# Multimodal Deep Learning for Public Safety Incident Detection

## Overview
This project presents a Transformer-based multimodal framework for violence detection in surveillance videos using RGB, Optical Flow, and Audio features.

The model processes each modality independently through Transformer Encoders and combines them using an attention-based fusion mechanism to improve violence detection performance.

## Features
- RGB Feature Analysis (I3D Features)
- Optical Flow Motion Analysis
- Audio Feature Analysis (VGGish Features)
- Transformer-Based Temporal Modeling
- Attention-Based Fusion
- Multi-Label Classification
- Weighted Binary Cross Entropy Loss

## Dataset
XD-Violence Dataset

Classes:
- Normal
- Fighting
- Shooting
- Riot
- Car Accident
- Explosion

## Architecture
1. RGB Features (I3D)
2. Optical Flow Features (I3D)
3. Audio Features (VGGish)
4. Transformer Encoders
5. Attention-Based Fusion Layer
6. Classification Layer

## Results
| Metric | Value |
|----------|----------
| mAP | 88.52% |
| Macro F1 Score | 0.81 |

## Technologies Used
- Python
- PyTorch
- NumPy
- Scikit-Learn
- Deep Learning
- Transformers
- Computer Vision

## License
MIT License
