# Day 28: Transformer Attention Visualizer and Embedding Inspector

An interpretability and diagnostic workbench designed to extract, visualize, and analyze multi-head self-attention maps and high-dimensional token embeddings from transformer language models.

## Overview

Understanding transformer mechanics requires inspecting the internal attention patterns that assign dynamic contextual weights between tokens. This project visualizes self-attention weights and applies dimensionality reduction to token embedding spaces.

## Architecture & Engineering Decisions

1. Attention Extraction: Hooks into PyTorch transformer layers (`output_attentions=True`) to extract per-head attention weight matrices across multiple transformer layers.
2. Embedding Dimensionality Reduction: Applies Principal Component Analysis (PCA) to project high-dimensional hidden state vectors onto interactive 2D coordinate projections.
3. Interactive Visualization: Generates heatmaps and Plotly scatter plots illustrating contextual token relationships and semantic clustering.

## Tech Stack

- Frameworks: PyTorch, Hugging Face Transformers
- Analytics & Dimensionality: NumPy, Scikit-learn (PCA)
- Visualization: Plotly, Matplotlib, Seaborn
- Interface & Serving: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day28-Transformer-Explorer.git
cd Day28-Transformer-Explorer
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day28-Transformer-Explorer)
