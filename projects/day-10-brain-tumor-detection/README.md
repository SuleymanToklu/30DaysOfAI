# Day 10: Brain Tumor Detection from MRI Scans

A parameter-efficient convolutional neural network pipeline for binary brain tumor classification from magnetic resonance imaging (MRI) scans.

## Overview

Medical image classification frequently suffers from severe data scarcity and high variance across patient scans. This project documents the architectural evolution from an over-parameterized baseline model to a compact, production-ready classifier designed for generalization on clinical scans.

## Architecture & Engineering Decisions

1. Baseline Diagnosis: An initial CNN baseline achieved 82% validation accuracy but relied on a dense `Flatten` layer that resulted in 5.4 million trainable parameters. Adding data augmentation and batch normalization caused severe accuracy degradation (30–60%), exposing acute memorization and overfitting.
2. Parameter Optimization: Replacing `Flatten` with `GlobalAveragePooling2D` reduced the parameter count by approximately 98% (from 5.4M to 110k parameters).
3. Regularization & Final Metric: Combined with spatial augmentations (rotations, zooms), the efficient model stabilized at 80% validation accuracy with significantly tighter train-validation loss convergence.

## Tech Stack

- Frameworks: TensorFlow / Keras, OpenCV, NumPy
- Model: Custom CNN with GlobalAveragePooling2D
- Serving & Interface: Streamlit Community Cloud

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day10-Brain-Tumor-Detection.git
cd Day10-Brain-Tumor-Detection
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day10-brain-tumor-detection-ueecw29psrpnfd9ybtpvbx.streamlit.app/)
