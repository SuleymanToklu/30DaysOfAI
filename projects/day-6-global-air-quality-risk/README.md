# Day 6: Global Air Quality and Environmental Health Risk Classifier

An environmental telemetry pipeline that queries real-time atmospheric data via API and predicts clinical health risk categories using an ensemble classifier.

## Overview

Air quality indices vary significantly by geographical geography, meteorological factors, and pollutant concentrations (PM2.5, PM10, O3, NO2). This application automates historical and live sensor data retrieval to deliver instant multi-class health risk assessments.

## Engineering & Modeling Decisions

1. Autonomous Telemetry Retrieval: Integrated Open-Meteo's atmospheric REST API, pulling historical pollutant and weather records on-demand without requiring manual file uploads.
2. Interactive Geographic Rendering: Integrated geospatial coordinates into Streamlit (`st.map`) alongside real-time pollutant metrics (`st.metric`) and transient toast notifications (`st.toast`).
3. Model Architecture: A Random Forest Classifier trained on pollutant thresholds, predicting Low, Moderate, and High respiratory risk levels with multilingual support (EN/TR).

## Tech Stack

- Languages & APIs: Python, Requests (Open-Meteo API)
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-learn (RandomForestClassifier), Joblib
- Frontend & Mapping: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day6-Global-Air-Quality-Risk.git
cd Day6-Global-Air-Quality-Risk
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day6-global-air-quality-risk-gjyuczch2shomvuiahjjdq.streamlit.app/)
