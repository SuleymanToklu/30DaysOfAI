# Day 19: Neural Text-to-Speech Synthesis

A natural language audio synthesis pipeline that converts raw text into natural, expressive speech waveforms with multi-accent and language support.

## Overview

Modern text-to-speech architectures generate natural prosody and acoustic nuances from text tokens. This project implements a clean web service for on-demand speech synthesis with modular voice profiles.

## Engineering & Modeling Decisions

1. Synthesis Pipeline: Evaluated speech generation backends for low-latency audio packet production and clear phoneme articulation.
2. Multi-Profile Selection: Supported parameterized controls for accent, pitch, and speech rate to adapt generated speech across different communication contexts.
3. Interactive Audio Rendering: Deployed with an interactive interface allowing instantaneous playback and raw `.wav` download.

## Tech Stack

- Core: Python, NumPy
- Audio & Synthesis: gTTS / edge-tts, Pydub
- Interface & Serving: Gradio / Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day-19-AI-Text-to-Speech-Converter.git
cd Day-19-AI-Text-to-Speech-Converter
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day-19-Text-to-Speech)
