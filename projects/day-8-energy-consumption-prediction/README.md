# Day 8: Smart Home Energy Consumption Benchmark Suite

A systematic comparative study and regression suite evaluating 7 distinct machine learning algorithms on IoT sensor time-series data for household appliance energy prediction.

## Overview

Smart home energy management requires models capable of learning from cyclical environmental data (temperature, humidity, pressure) without overreacting to transient noise. This project conducts an empirical benchmark across linear, regularized, support vector, and tree ensemble models.

## Engineering & Modeling Decisions

1. Algorithmic Benchmarking: Evaluated 7 estimators under identical cross-validation conditions:
   - Linear Regression
   - Ridge Regression (L2)
   - Lasso Regression (L1)
   - Support Vector Regressor (SVR)
   - Random Forest Regressor
   - XGBoost Regressor
   - LightGBM Regressor
2. Cyclical Feature Extraction: Decomposed raw timestamps into `hour`, `day_of_week`, and `month` components to capture periodic consumption cycles.
3. Side-by-Side Visualization: Integrated Altair multi-model comparison charts within Streamlit, rendering live predictions from all 7 models concurrently.

## Tech Stack

- Core: Python, Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost, LightGBM
- Visualization & Web: Streamlit, Altair

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day8-Energy-Consumption-Prediction.git
cd Day8-Energy-Consumption-Prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day8-energy-consumption-prediction-5enokgupsuatgd3crwke9a.streamlit.app/)
