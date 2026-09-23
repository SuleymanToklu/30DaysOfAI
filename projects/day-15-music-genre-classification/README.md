# Day 15: Audio Spectrogram Feature Extraction and Music Genre Classification

An audio machine learning pipeline that extracts spectral descriptors from raw acoustic waveforms and classifies tracks using high-dimensional Support Vector Classifiers.

## Overview

Acoustic classification requires transforming non-stationary audio signals into structured frequency representations that capture timbre, rhythm, and harmonic content. This project processes raw `.wav` audio files to classify tracks across standard musical genres.

## Engineering & Modeling Decisions

1. Spectral Feature Engineering: Employed `librosa` to compute:
   - Mel-Frequency Cepstral Coefficients (MFCCs)
   - Spectral Centroid & Spectral Rolloff
   - Zero-Crossing Rate (ZCR)
   - Chroma Short-Time Fourier Transforms (STFT)
2. Statistical Aggregation: Extracted summary statistics (mean, variance) across audio time frames to produce compact, fixed-width feature vectors.
3. Classification & Serving: Trained a Support Vector Machine (RBF kernel) optimized for high-dimensional feature spaces, served through a Streamlit audio analysis dashboard.

## Tech Stack

- Core & Audio: Python, Librosa, NumPy, Pandas
- Machine Learning: Scikit-learn (SVC), Joblib
- Interface: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day15-Music-Genre-Classification.git
cd Day15-Music-Genre-Classification
pip install -r requirements.txt
python process_and_train.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day15-music-genre-classification-clcay5suykzd2cmh5lgerf.streamlit.app/)
