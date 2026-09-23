import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

def create_risk_level(aqi):
    """Creates a risk category based on the overall AQI value."""
    if aqi <= 50:
        return 0  
    elif aqi <= 100:
        return 1  
    else:
        return 2  

def run_training_pipeline():
    """
    Loads global air pollution data, trains the model, and saves all artifacts.
    """
    print("--- Training Pipeline Started ---")

    print("1/4 - Loading data...")
    try:
        df = pd.read_csv("global air pollution.csv")
    except FileNotFoundError:
        print("HATA: 'global air pollution.csv' bulunamadı.")
        return

    print("2/4 - Preprocessing data...")
    df['Risk_Level'] = df['AQI Value'].apply(create_risk_level)
    le = LabelEncoder()
    df['City'] = df['City'].astype(str)
    df['City_Encoded'] = le.fit_transform(df['City'])
    
    print("--- Creating country-city map ---")
    country_city_map = df.groupby('Country')['City'].unique().apply(list).to_dict()

    features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'City_Encoded']
    X = df[features]
    y = df['Risk_Level']
    
    print("3/4 - Training RandomForestClassifier model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
    model.fit(X, y)

    print("4/4 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'model_features.pkl')
    joblib.dump(le, 'city_encoder.pkl')
    joblib.dump(country_city_map, 'country_city_map.pkl') 
    
    print("--- Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_training_pipeline()
