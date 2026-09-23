import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_processing_pipeline():
    """
    Loads news headlines, processes them, trains an LDA topic model,
    and saves the necessary artifacts.
    """
    print("--- Data Processing Pipeline Started ---")

    print("1/3 - Loading data...")
    try:
        df = pd.read_csv("abcnews-date-text.csv", parse_dates=['publish_date'])
    except FileNotFoundError:
        print("ERROR: 'abcnews-date-text.csv' not found. Make sure it's in the same directory.")
        return

    df_sample = df.sample(n=20000, random_state=42)
    
    print("2/3 - Vectorizing text and training LDA model...")

    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df_sample['headline_text'])

    n_topics = 10
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)
    
    print("3/3 - Saving the processed artifacts...")
    joblib.dump(lda_model, 'lda_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("--- Data Processing Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_processing_pipeline()