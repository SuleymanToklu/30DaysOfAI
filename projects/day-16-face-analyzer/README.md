---
title: Face Analyzer
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---

# Day 16: Facial Attribute and Demographic Analysis

A deep learning facial perception pipeline utilizing DeepFace backbones for automated facial landmark detection, demographic estimation, and emotion classification.

## Overview

Facial analysis systems combine face detection with multi-task convolutional neural networks to predict attributes including estimated age, gender classification, dominant emotional state, and demographic cues from digital images.

## Architecture & Engineering Decisions

1. Modular Backbone Architecture: Utilizes the `deepface` framework with TensorFlow execution backends for robust bounding box detection under non-cooperative lighting and varied face orientations.
2. Multi-Task Attribute Estimation: Processes isolated facial crops through pre-trained networks to infer emotional expressions and demographic markers.
3. Gradio Interface: Deployed on Hugging Face Spaces with drag-and-drop image upload and structured JSON/text output rendering.

## Tech Stack

- Deep Learning & Vision: TensorFlow, DeepFace, OpenCV
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/day-16-face-analyzer.git
cd day-16-face-analyzer
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/day-16-face-analyzer)
