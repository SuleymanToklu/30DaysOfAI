# Day 12: Hybrid Movie Recommender System

A hybrid recommendation engine combining content-based natural language filtering with matrix-factorized collaborative filtering.

## Overview

Pure content-based systems suffer from limited serendipity, while pure collaborative filtering breaks down on new items (the cold-start problem). This system implements a hybrid architecture that synergizes text similarity with latent factor interaction modeling.

## Architecture & Engineering Decisions

1. Content-Based Subsystem: Utilizes TF-IDF vectorization across metadata corpora (genres, cast, plot overviews) with cosine similarity ranking.
2. Collaborative Filtering Subsystem: Implements Singular Value Decomposition (SVD) via the `scikit-surprise` library to discover latent user-item interaction vectors.
3. Hybrid Fusion: Blends normalized similarity scores with predicted user ratings to rank suggestions, handling both cold-start and established-profile scenarios.

## Tech Stack

- Core: Python, Pandas, NumPy
- Machine Learning: Scikit-learn (TF-IDF), Scikit-surprise (SVD)
- Web Interface: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day12-Movie-Recommender-System.git
cd Day12-Movie-Recommender-System
pip install -r requirements.txt
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day12-movie-recommender-system.streamlit.app/)
