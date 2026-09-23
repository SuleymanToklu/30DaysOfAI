import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score
import joblib

print("Loading dataset...")
df = pd.read_csv('projectile_dataset.csv')

features = ['target_x', 'target_y']
labels = ['initial_velocity', 'launch_angle']

X = df[features]
y = df[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- YENİ ÖLÇEKLEME KISMI ---
print("Scaling data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Ölçekleyicileri sadece eğitim verisiyle eğitiyoruz
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Test verisini de aynı ölçekleyiciyle dönüştürüyoruz
X_test_scaled = scaler_X.transform(X_test)
# --- ÖLÇEKLEME KISMI SONU ---

print(f"Training model on {len(X_train)} scaled samples...")

model = MLPRegressor(
    hidden_layer_sizes=(256, 512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    verbose=True,
    early_stopping=True,
    n_iter_no_change=30
)

# Modeli ÖLÇEKLENMİŞ veriyle eğitiyoruz
model.fit(X_train_scaled, y_train_scaled)

print("Training complete.")

predictions_scaled = model.predict(X_test_scaled)
# Performansı değerlendirmek için tahminleri geri ölçekliyoruz
predictions_real = scaler_y.inverse_transform(predictions_scaled)
score = r2_score(y_test, predictions_real)
print(f"Model R^2 score on test data: {score:.4f}")

# Sadece modeli değil, ölçekleyicileri de bir paket olarak kaydediyoruz
model_bundle = {
    'model': model,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y
}
joblib.dump(model_bundle, 'projectile_model.joblib')
print("Model and scalers saved as a bundle in projectile_model.joblib")