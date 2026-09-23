# Day 5: Financial Stock Volatility Forecasting

A machine learning time-series regression pipeline designed to predict equity price volatility using dynamic market data and rolling feature engineering.

## Overview

Volatility is a critical input for portfolio risk management, option pricing, and algorithmic hedging. This project models future price dispersion by combining historical market indicators fetched dynamically via API.

## Engineering & Modeling Decisions

1. Live Data Ingestion: Employed `yfinance` to stream historical daily market records for arbitrary tickers, eliminating reliance on static local CSV archives.
2. Temporal Feature Engineering: Constructed lag features, rolling moving averages, exponential moving averages, and historical return variance windows without look-ahead bias.
3. Gradient Boosting Regression: Trained an XGBoost Regressor tuned to capture non-linear market shocks while preventing autoregressive overfitting.

## Tech Stack

- Data Ingestion: yfinance API
- Processing & Feature Engineering: Pandas, NumPy
- Predictive Modeling: XGBoost, Scikit-learn
- Application Framework: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day5-Stock-Volatility-Prediction.git
cd Day5-Stock-Volatility-Prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day5-stock-volatility-prediction-yrwywwvwfd9nsqdgsefr3x.streamlit.app/)
