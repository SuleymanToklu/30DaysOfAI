# Day 20: Real-Time Handwritten Digit Recognition

An interactive machine learning application that processes user-drawn digital canvas sketches and classifies digits (0-9) using Support Vector Classifiers.

## Overview

Optical character and digit recognition demonstrates foundational machine learning concepts: mapping raw pixel arrays to discrete semantic categories. This project provides an interactive digital drawing canvas for real-time digit recognition.

## Engineering & Modeling Decisions

1. Preprocessing Pipeline: Converted freehand user drawings into 28x28 grayscale matrices with center-of-mass normalization, mirroring standard MNIST preprocessing.
2. Classifier Architecture: Trained a Support Vector Classifier (SVC) using Scikit-learn with RBF kernels to handle non-linear pixel boundary distributions.
3. Gradio Sketchpad: Integrated Gradio's `Sketchpad` input component to allow direct freehand mouse/stylus drawing with instant inference response.

## Tech Stack

- Core: Python, NumPy, Pillow
- Modeling: Scikit-learn (SVC), Joblib
- Interface: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day20-digit-recognizer.git
cd Day20-digit-recognizer
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day20-digit-recognizer)
