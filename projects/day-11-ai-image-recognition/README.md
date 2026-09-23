# Day 11: Transfer Learning Image Classification

A computer vision classification pipeline utilizing pre-trained deep convolutional neural networks for multi-class object recognition via transfer learning.

## Overview

Training deep visual backbones from scratch requires millions of annotated samples and extensive compute. Transfer learning leverages rich hierarchical representations pre-trained on ImageNet to achieve high-accuracy inference with minimal fine-tuning overhead.

## Engineering & Modeling Decisions

1. Backbone Selection: Evaluated lightweight architectures (MobileNetV2 / ResNet) to balance inference latency with top-1 classification accuracy.
2. Feature Reuse: Frozen convolutional feature extractors coupled with custom classification heads (Dropout, Dense, Softmax) to adapt general feature maps to target categories.
3. Streamlit Interface: Developed a clean interface allowing real-time image uploads, top-K confidence visualizers, and inference timing metrics.

## Tech Stack

- Frameworks: Python, TensorFlow / Keras, NumPy, Pillow
- Serving: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day11-AI-Image-Recognition.git
cd Day11-AI-Image-Recognition
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day11-ai-image-recognition.streamlit.app/)
