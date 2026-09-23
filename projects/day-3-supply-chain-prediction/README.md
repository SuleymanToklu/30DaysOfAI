# Day 3: Hyperion - Dynamic Supply Chain Delay Prediction

A predictive machine learning system designed to forecast logistics and delivery delays across complex supply chains by fusing internal shipment logs with real-time external telemetry.

## Overview

Static shipping estimations often fail due to unforeseen regional disruptions, market volatility, and seasonal calendar anomalies. Project Hyperion integrates multi-source feature extraction to produce robust delivery delay forecasts.

## Engineering & Modeling Decisions

1. External Telemetry Enrichment: Integrated real-time market signals (`yfinance` fuel and index proxies), public holiday schedules (`holidays`), and geographic parameters via REST APIs.
2. Resilient Gradient Boosting: LightGBM was chosen for its high execution speed, native handling of categorical features, and robustness to outliers in shipping durations.
3. Production Dashboard: Built a Streamlit control center with interactive Plotly visualizers for scenario testing and sensitivity analysis.

## Tech Stack

- Core: Python, Pandas, NumPy
- Modeling: LightGBM, Scikit-learn
- Ingestion & Signals: Requests, Holidays, yfinance
- Visualization & UI: Streamlit, Plotly

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day3-Supply-Chain-Prediction.git
cd Day3-Supply-Chain-Prediction
pip install -r requirements.txt
python build_features.py
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day3-supply-chain-prediction-fsjs3hfxayj7yftuq366u5.streamlit.app/)
