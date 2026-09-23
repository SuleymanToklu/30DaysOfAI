# Day 2: E-Commerce Shopping Cart Abandonment Prediction

A binary classification system engineered to identify high-risk session abandonment before checkout, enabling proactive retention workflows.

## Overview

In e-commerce analytics, standard accuracy is an inadequate metric due to severe class imbalance: only ~15% of user sessions culminate in a purchase. This project develops an end-to-end production pipeline emphasizing class-weighted recall over naive accuracy.

## Engineering & Modeling Decisions

1. Metric Alignment & Cost-Sensitive Learning: Naive models achieved high accuracy by predicting the majority class (non-purchase). To minimize missed conversion opportunities, optimization was focused on Recall, leveraging XGBoost's `scale_pos_weight` hyperparameter to penalize false negatives.
2. Decoupled Pipeline Architecture: Training, feature engineering, and serialization were separated into a standalone script (`train_model.py`) rather than a notebook, outputting verified model artifacts for isolated inference.
3. Production Serving: The serialized pipeline is served via Streamlit with real-time probability threshold tuning.

## Tech Stack

- Core: Python, Pandas, NumPy
- Machine Learning: XGBoost, Scikit-learn
- Pipeline & Serialization: Joblib
- Interface: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day2-Cart-Abandonment-Final.git
cd Day2-Cart-Abandonment-Final
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day2-cart-abandonment-final-gyt7thfrhdag8g6k5w2z8h.streamlit.app/)
