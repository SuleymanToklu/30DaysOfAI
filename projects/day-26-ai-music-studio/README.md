---
title: AI Music Studio
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 26: Conditional Audio Synthesis with MusicGen

A generative audio engineering pipeline implementing Meta's MusicGen architecture to synthesize original acoustic and musical sequences from natural language descriptions.

## Overview

Conditional audio generation models autoregressively predict audio tokens over discrete acoustic codebooks. This project deploys Meta's MusicGen model in a production web studio for parameter-controlled music synthesis.

## Architecture & Engineering Decisions

1. Model Architecture: Implements `facebook/musicgen-small`, utilizing an EnCodec audio tokenizer and an autoregressive transformer to generate stereo audio tokens.
2. Generation Parameter Controls: Exposes duration controls, guidance scale tuning, and sampling temperatures to control harmonic density and adherence to text prompts.
3. Audio Serialization: Employs `scipy.io.wavfile` to write raw 32 kHz floating-point waveforms into lossless `.wav` containers for in-browser playback.

## Tech Stack

- Deep Learning & Audio: PyTorch, Transformers, Scipy, Torchaudio
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day26-AI-Music-Studio.git
cd Day26-AI-Music-Studio
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day-26-AI-Music-Studio)
