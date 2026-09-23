# Day 14: Real-Time Object Detection with YOLOv5

A high-throughput computer vision pipeline utilizing YOLOv5 for single-pass bounding box prediction and multi-class object localization.

## Overview

Real-time vision systems require unified object detection architectures that predict bounding boxes and class probabilities in a single forward pass, avoiding the latency of two-stage region proposal networks.

## Engineering & Modeling Decisions

1. Pre-Trained Weights via PyTorch Hub: Leveraged official YOLOv5s checkpoint weights, optimizing model load times and memory footprint for edge and cloud deployment.
2. Inference Pipeline: Integrated OpenCV image decoding with Ultralytics non-maximum suppression (NMS) to eliminate duplicate detections.
3. Multilingual Interface: Developed an interactive Streamlit UI with bilingual support (TR/EN) and adjustable confidence/IOU threshold sliders.

## Tech Stack

- Core: Python, OpenCV, NumPy
- Deep Learning: PyTorch, YOLOv5 (Ultralytics)
- Serving: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day14-YOLO-Object-Detection.git
cd Day14-YOLO-Object-Detection
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day14-yolo-object-detection-9x3wa8xrzfzy3bhkvb7csf.streamlit.app/)
