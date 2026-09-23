import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def extract_features(file_path):
    # Extracts audio features from a given .wav file.
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_means = [np.mean(e) for e in mfcc]
        
        features = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr] + mfcc_means
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def run_training_pipeline():
    # Processes audio files, trains a classification model, and saves artifacts.
    print("--- Training Pipeline Started ---")
    DATA_DIR = "genres_original"
    
    # 1. Feature Extraction
    print("1/3 - Extracting audio features (this will take several minutes)...")
    all_features = []
    all_labels = []
    genres = [g for g in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, g))]

    for genre in tqdm(genres, desc="Processing genres"):
        genre_path = os.path.join(DATA_DIR, genre)
        for filename in os.listdir(genre_path):
            file_path = os.path.join(genre_path, filename)
            features = extract_features(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(genre)

    # 2. Prepare for Modeling
    print("2/3 - Preparing data and training SVM model...")
    X = np.array(all_features)
    y = np.array(all_labels)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    # 3. Save Artifacts
    print("3/3 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    
    print("--- Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_training_pipeline()