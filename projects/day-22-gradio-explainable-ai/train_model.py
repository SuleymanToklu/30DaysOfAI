import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import shap

def train_and_save():
    """
    Trains the heart disease model and saves the necessary files (.pkl).
    This function will be called by the app on its first run if models are not found.
    """
    print("Model files not found. Starting training process...")
    
    # Load the dataset
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(url, header=None, names=column_names, na_values='?')
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # Data Preprocessing
    df.dropna(inplace=True)
    df['target'] = (df['target'] > 0).astype(int)
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save Model, Explainer and Feature Names
    joblib.dump(model, 'heart_disease_model.pkl')
    print("Model saved to heart_disease_model.pkl")

    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, 'shap_explainer.pkl')
    print("SHAP explainer saved to shap_explainer.pkl")

    joblib.dump(feature_names, 'feature_names.pkl')
    print("Feature names saved to feature_names.pkl")
    
    print("Training complete. Models are ready.")

# This allows the script to be run directly for local testing if needed
if __name__ == '__main__':
    train_and_save()

