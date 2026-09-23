---
title: Image Storyteller
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 25: Multimodal Image-to-Story Generation Pipeline

An end-to-end multimodal pipeline that takes an uploaded image, generates descriptive semantic captions, and expands the scene into a structured narrative using autoregressive language models.

## Overview

Connecting computer vision with creative text generation requires sequential multimodal composition: vision backbones extract concrete scene semantics, which subsequently serve as context prompts for generative language models.

## Architecture & Engineering Decisions

1. Vision-Language Ingestion: Uses BLIP (`Salesforce/blip-image-captioning-large`) to produce an initial factual description of the input image.
2. Narrative Expansion: Feeds the extracted visual prompt into an autoregressive language model configured with temperature sampling to generate coherent, creative prose.
3. Web Interface: Deployed as an interactive application on Hugging Face Spaces.

## Tech Stack

- Deep Learning: PyTorch, Transformers, Pillow
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day25-AI-Image-Storyteller.git
cd Day25-AI-Image-Storyteller
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day25-AI-Image-Storyteller)
