# Day 7: Acoustic Song Popularity Regression

A regression engine that evaluates Spotify audio signal features and metadata to forecast track popularity scores.

## Overview

Music streaming platforms rely heavily on audio descriptors to structure recommendation networks and evaluate release potential. This project implements a gradient boosted regressor trained on acoustic signal metrics to estimate audience reach.

## Engineering & Modeling Decisions

1. Domain Feature Extraction: Modeled multidimensional acoustic metrics including danceability, acousticness, energy, valence, instrumentalness, and tempo.
2. Model Training & Serialization: Utilized XGBoost Regressor with hyperparameter tuning, serializing both the trained estimator and the feature schema into decoupled Joblib artifacts.
3. Interactive Testing: Built a Streamlit application allowing users to manipulate audio metrics and examine the resulting predicted popularity scores.

## Tech Stack

- Core: Python, Pandas, NumPy
- Modeling & Pipelines: Scikit-learn, XGBoost, Joblib
- Interface: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day7-Song-Popularity-Prediction.git
cd Day7-Song-Popularity-Prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day7-song-popularity-prediction-eexedi4he5tbuappcrhwbax.streamlit.app/)
