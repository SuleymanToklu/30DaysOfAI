---
title: Dynamic Classifier Trainer
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Day 27: Interactive Machine Learning Classifier Trainer

An interactive training and experimentation workbench that enables dynamic dataset ingestion, automated preprocessing, real-time hyperparameter configuration, and classification model evaluation.

## Overview

Experimentation in machine learning often requires rapid iteration between algorithmic families and validation metrics. This project implements a self-contained AutoML training studio within Gradio.

## Architecture & Engineering Decisions

1. Modular Estimator Engine: Supports dynamic model selection across:
   - Logistic Regression
   - Support Vector Machines (Linear & RBF)
   - Decision Trees & Random Forests
   - Gradient Boosting (XGBoost / LightGBM)
2. Automated Preprocessing Pipeline: Integrates automated missing-value imputation, categorical encoding, and feature scaling (`StandardScaler` / `MinMaxScaler`).
3. Evaluation & Metrics: Generates confusion matrices, classification reports (precision, recall, F1), and ROC curves dynamically upon training completion.

## Tech Stack

- Machine Learning & Data: Python, Pandas, NumPy, Scikit-learn
- Visualization: Matplotlib, Seaborn
- Deployment: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day-27-AI-Classifier-Trainer.git
cd Day-27-AI-Classifier-Trainer
pip install -r requirements.txt
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day-27-AI-Classifier-Trainer)
