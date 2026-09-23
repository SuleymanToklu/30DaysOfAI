# Day 21: Abstractive Text Summarization with Sequence-to-Sequence Transformers

An abstractive summarization service powered by Hugging Face Transformers, generating concise, context-preserving summaries from long-form documents.

## Overview

Unlike extractive summarization which merely subsets existing sentences, abstractive summarization synthesizes novel sentences that capture the essential semantic meaning of long texts.

## Architecture & Engineering Decisions

1. Sequence-to-Sequence Modeling: Deployed pre-trained transformer checkpoints (e.g. `facebook/bart-large-cnn` or T5) optimized for sequence-to-sequence summarization tasks.
2. Generation Parameter Tuning: Configured beam search decoding with length penalties and repetition penalties to prevent degenerative looping while maintaining factual fidelity.
3. Web Interface: Built a responsive Gradio interface deployed on Hugging Face Spaces supporting arbitrary document input and configurable summary length bounds.

## Tech Stack

- Deep Learning & NLP: PyTorch, Hugging Face Transformers
- Interface & Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day-21-Summarizer-Gradio.git
cd Day-21-Summarizer-Gradio
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day-21-Summarizer-Gradio)
