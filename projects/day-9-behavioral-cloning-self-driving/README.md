# Day 9: Autonomous Vehicle Road Scene Recognition

A deep learning vision pipeline designed to recognize road attributes, track conditions, and environmental context for autonomous driving applications.

## Overview

Perception layers in autonomous vehicles must identify road geometries, boundaries, and obstacles under variable lighting and perspective shifts. This project focuses on building and evaluating convolutional feature extractors for automotive scene understanding.

## Engineering & Modeling Decisions

1. Convolutional Feature Extraction: Designed a multi-stage CNN architecture utilizing spatial pooling and dropout regularization to extract high-level road geometry representations.
2. Data Preprocessing & Augmentation: Implemented custom image normalizations, horizontal flips, and contrast adjustments to build resilience against shadow artifacts and lens flare.
3. Streamlit Deployment: Built an interactive inspection dashboard that displays classification probabilities and layer activation summaries.

## Tech Stack

- Core: Python, OpenCV, NumPy
- Deep Learning: PyTorch / TensorFlow, Keras
- Serving: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day9-Behavioral-Cloning-Self-Driving.git
cd Day9-Behavioral-Cloning-Self-Driving
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day9-behavioral-cloning-self-driving-hh3bfzlvnybsvxwqwotlui.streamlit.app/)
