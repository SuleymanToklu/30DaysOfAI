# Day 4: Telco Customer Churn Prediction

A customer churn classification engine engineered to predict subscription attrition risks and surface actionable feature attributions for proactive retention.

## Overview

Customer acquisition costs substantially exceed retention expenses in the telecommunications industry. This project implements a calibrated classification pipeline capable of scoring churn probabilities across varying contract types, service bundles, and billing profiles.

## Engineering & Modeling Decisions

1. Imbalanced Class Weighting: Addressed class imbalance using XGBoost's `scale_pos_weight` parameter to ensure the loss function prioritizes minority churn cases without synthetic oversampling artifacts.
2. Interactive Scenario Simulation: Designed a Streamlit control dashboard allowing customer success operators to manipulate account attributes and observe instantaneous risk profile adjustments.
3. Feature Importance Analysis: Evaluated feature gains to identify high-risk contract types (e.g. month-to-month contracts and fiber optic without tech support).

## Tech Stack

- Core: Python, Pandas, NumPy
- Machine Learning: XGBoost, Scikit-learn
- Web Interface: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day4-Customer-Churn-Prediction.git
cd Day4-Customer-Churn-Prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day4-customer-churn-prediction-jjywstjjj3l252e39v39ea.streamlit.app/)
