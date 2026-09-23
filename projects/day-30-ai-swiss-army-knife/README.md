---
title: AI Swiss Army Knife
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: true
---

# Day 30: AI Swiss Army Knife Integrated Suite

A unified multi-modal production application integrating latent diffusion text-to-image synthesis, vision-language image captioning, and speech-to-text transcription within a single multi-tabbed interface.

## Overview

The final capstone of the 30 Days of AI marathon consolidates three foundational deep learning modalities into an integrated, modular web suite:
1. Generative Vision: Latent diffusion text-to-image synthesis (`CompVis/ldm-text2im-large-256`).
2. Vision-Language Understanding: Automated image captioning via BLIP (`Salesforce/blip-image-captioning-large`).
3. Acoustic Speech Perception: Automatic speech recognition and transcription powered by OpenAI's Whisper (`openai/whisper-base`).

## Architecture & Engineering Decisions

1. Modular Pipeline Initialization: Encapsulates each model family in lazy-loaded execution pipelines to manage GPU/CPU memory allocation efficiently.
2. Unified Interface Architecture: Implements a clean, tabbed Gradio layout allowing seamless switching between modalities without session state corruption.
3. Production Serving: Deployed on Hugging Face Spaces with robust error handling and structured output presentation.

## Tech Stack

- Frameworks: PyTorch, Hugging Face (Diffusers, Transformers)
- Models: Latent Diffusion, BLIP, OpenAI Whisper
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day30-AI-Swiss-Army-Knife.git
cd Day30-AI-Swiss-Army-Knife
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day30-AI-Swiss-Army-Knife)
