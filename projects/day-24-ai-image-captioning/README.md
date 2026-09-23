---
title: Image Captioning Studio
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 24: Multimodal Vision-Language Image Captioning

A multimodal deep learning system that takes arbitrary visual inputs and produces accurate natural language descriptions using the BLIP architecture.

## Overview

Vision-language models bridge visual perception and language understanding by mapping image feature representations into text generation decoders. This project implements the BLIP (Bootstrapping Language-Image Pre-training) architecture for zero-shot image captioning.

## Architecture & Engineering Decisions

1. Model Backbone: Utilizes `Salesforce/blip-image-captioning-large`, integrating a vision transformer image encoder with a text decoder.
2. Beam Search Decoding: Configured beam width and repetition penalties during auto-regressive generation to produce coherent, detailed descriptions without repetitive phrases.
3. Interface: Deployed on Hugging Face Spaces with drag-and-drop image uploads and instant caption generation.

## Tech Stack

- Deep Learning & Multimodal: PyTorch, Transformers, Pillow
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day24-AI-Image-Captioning.git
cd Day24-AI-Image-Captioning
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day24-AI-Image-Captioning)
