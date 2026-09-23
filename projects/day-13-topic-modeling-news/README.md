# Day 13: Unsupervised Topic Modeling on News Headlines

An unsupervised natural language processing pipeline utilizing Latent Dirichlet Allocation (LDA) to extract hidden thematic structures from large-scale news headline corpora.

## Overview

Processing unstructured text streams at scale requires statistical topic extraction to cluster documents without manual labeling. This project implements an LDA generative statistical model trained on the Million News Headlines dataset.

## Engineering & Modeling Decisions

1. Text Preprocessing Pipeline: Built tokenization, stop-word elimination, length filtering, and lemmatization routines to clean noisy headline text.
2. Vocabulary Pruning: Evaluated document-frequency thresholds (`min_df`, `max_df`) with CountVectorizer to eliminate rare artifacts and universally distributed vocabulary.
3. Latent Dirichlet Allocation: Decomposed term-document matrices into latent topic distributions, enabling interactive keyword exploration via Streamlit.

## Tech Stack

- Core & NLP: Python, Pandas, Scikit-learn (CountVectorizer, LatentDirichletAllocation)
- Serving: Streamlit

## Setup & Execution

```bash
git clone https://github.com/SuleymanToklu/Day13-Topic-Modeling-News.git
cd Day13-Topic-Modeling-News
pip install -r requirements.txt
python process_data.py
streamlit run app.py
```

## Production Deployment

- Live Application: [Streamlit Cloud](https://day13-topic-modeling-news-jws5mzzyix6kqjrddvnc69.streamlit.app/)
