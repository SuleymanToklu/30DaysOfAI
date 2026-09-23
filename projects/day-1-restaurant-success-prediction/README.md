# Day 1: Restaurant Success Score Predictor

A predictive regression model that estimates a restaurant's commercial success score based on operational attributes including location, cuisine diversity, cost tier, and service availability.

## Overview

Predicting business viability in the hospitality sector requires modeling non-linear interactions across spatial and operational variables. This project establishes an end-to-end regression pipeline from raw tabular data to a cloud-deployed inference interface.

## Engineering & Modeling Decisions

1. Handling Constrained Environments: The initial raw dataset exceeded 500 MB, introducing significant memory overhead and cloud deployment timeouts. The data pipeline was refactored with stratified random sampling to preserve feature distribution while shrinking disk and RAM usage to production limits.
2. Artifact Management: Addressed Git LFS pointer tracking and remote server filesystem resolution during deployment to ensure zero-overhead container startups.
3. Model Selection: A Random Forest Regressor was trained on preprocessed tabular features, capturing feature interactions between geographic zone and cuisine pricing without requiring manual polynomial expansions.

## Tech Stack

- Languages & Core: Python, NumPy, Pandas
- Modeling & Validation: Scikit-learn (RandomForestRegressor)
- Visualization: Seaborn, Matplotlib
- Interface & Serving: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day1-Restaurant-Success-Prediction.git
cd Day1-Restaurant-Success-Prediction
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://30-days-30-projects-gj5pcrv8z8fo5n6umq3tq5.streamlit.app/)
