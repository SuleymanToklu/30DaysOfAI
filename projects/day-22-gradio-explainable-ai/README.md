---
title: Explainable AI Studio
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 22: Explainable AI (XAI) Model Interpretation

An interpretability workbench implementing SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to decompose black-box machine learning predictions.

## Overview

High-stakes machine learning deployments demand algorithmic transparency. This project implements attribution methods to explain individual model predictions and quantify global feature contributions.

## Architecture & Engineering Decisions

1. Shapley Additive Explanations (SHAP): Computes game-theoretic Shapley values to measure the marginal contribution of each feature to individual inference outputs.
2. Local Interpretable Model-Agnostic Explanations (LIME): Fits local surrogate linear models around specific prediction instances to explain complex decision boundaries.
3. Interactive Visualization: Renders waterfall, summary, and force plots directly within a Gradio interface for transparent inspection of model behavior.

## Tech Stack

- Core & ML: Python, Scikit-learn, NumPy, Pandas
- Interpretability: SHAP, LIME
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day22-Gradio-Explainable-AI.git
cd Day22-Gradio-Explainable-AI
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day22-Gradio-Explainable-AI)
