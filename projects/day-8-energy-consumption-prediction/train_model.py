import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_training_pipeline():
    """
    Loads data, splits it, engineers features, trains models, evaluates them,
    and saves all artifacts (models and metrics).
    """
    print("--- Training Pipeline Started ---")

    print("1/6 - Loading data...")
    df = pd.read_csv("energydata_complete.csv")

    print("2/6 - Engineering features...")
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    print("3/6 - Preparing and splitting data...")
    features = [
        'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
        'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 
        'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
        'hour', 'day_of_week', 'month'
    ]
    target = 'Appliances'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "SVR": SVR(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1)
    }

    print("4/6 - Training models...")
    for name, model in models.items():
        print(f"  - Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f'model_{name.replace(" ", "_")}.pkl')
    
    print("5/6 - Evaluating models and saving metrics...")
    metrics = {}
    for name, model in models.items():
        print(f"  - Evaluating {name}...")
        predictions = model.predict(X_test) 
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    
    joblib.dump(metrics, 'model_metrics.pkl')
    
    print("6/6 - Saving feature list...")
    joblib.dump(features, 'model_features.pkl')
    
    print("\n--- Training Pipeline Completed Successfully! ---")
    print(f"All {len(models)} models and their performance metrics have been saved.")

if __name__ == "__main__":
    run_training_pipeline()