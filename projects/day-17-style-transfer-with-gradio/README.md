---
title: Neural Style Transfer
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 17: Neural Style Transfer with VGG-19

A generative computer vision pipeline implementing the Gatys et al. Neural Style Transfer algorithm to synthesize images by blending content geometry with artistic textural representations.

## Overview

Neural Style Transfer operates as an optimization process where intermediate representations of a deep convolutional network separate and recombine image content and artistic style.

## Architecture & Engineering Decisions

1. Feature Extraction Backbone: Employs a pre-trained VGG-19 network with frozen parameters to extract content and style representations from specific convolutional activation maps.
2. Gram Matrix Style Formulation: Captures texture and color correlations across feature maps using Gram matrix outer products, establishing style representations invariant to spatial positioning.
3. Loss Optimization: Iteratively optimizes a target image by minimizing a composite loss function balancing mean squared content error with style Gram matrix discrepancy.
4. Serving: Encapsulated in a Gradio interface on Hugging Face Spaces with configurable style weight controls.

## Tech Stack

- Frameworks: PyTorch, Torchvision, NumPy, Pillow
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day17-Style-Transfer-with-Gradio.git
cd Day17-Style-Transfer-with-Gradio
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day17-Style-Transfer-with-Gradio)
