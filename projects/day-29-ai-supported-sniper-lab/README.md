---
title: Ballistics Trajectory Simulator and Inversion Engine
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# Day 29: AI-Assisted Ballistics Trajectory Simulator and Inversion Engine

An applied scientific machine learning application that pairs classical kinematic simulation with neural network regression to solve the inverse projectile trajectory problem: determining the initial velocity and launch angle required to strike target coordinates.

## Overview

Solving projectile motion in the forward direction is governed by standard kinematic equations. However, determining the inverse parameters required to hit arbitrary 2D target coordinates subject to physics constraints is non-trivial. This system simulates physics trajectories to train a neural regressor capable of instantaneous inverse parameter prediction.

## Architecture & Engineering Decisions

1. Synthetic Physics Trajectory Generation (`generate_data.py`):
   - Computes kinematic trajectory paths across diverse initial launch parameters (velocity, angle, gravitational acceleration).
   - Samples intermediate coordinate points along valid trajectories to assemble an inverse parameter training dataset.
2. Neural Network Inversion (`train_model.py`):
   - Features and targets are normalized using Scikit-learn `StandardScaler` instances to ensure numerical stability.
   - Trains a Multi-Layer Perceptron Regressor (`MLPRegressor`) to map target spatial coordinates `(x, y)` back to continuous launch parameters `[velocity, angle]`.
   - Bundles the trained network and scaling artifacts into a standalone serialized `.joblib` package.
3. Interactive Simulation Dashboard (`app.py`):
   - Accepts arbitrary coordinate targets, computes normalized model inferences, denormalizes the outputs, and plots the verified trajectory path via Matplotlib.

## Tech Stack

- Core & Simulation: Python, NumPy, Matplotlib
- Machine Learning: Scikit-learn (MLPRegressor, StandardScaler), Joblib
- Interface & Serving: Gradio, Hugging Face Spaces

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day-29-AI-Supported-Sniper-Lab.git
cd Day-29-AI-Supported-Sniper-Lab
pip install -r requirements.txt
python generate_data.py
python train_model.py
python app.py
```

## Production Deployment

- Live Application: [Hugging Face Space](https://huggingface.co/spaces/tiheli/Day-29-Fizik-Simulator)
