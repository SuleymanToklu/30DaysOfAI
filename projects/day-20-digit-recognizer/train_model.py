from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


print("Fetching the MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
print("MNIST dataset loaded successfully.")


y = y.astype(np.uint8)

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

print(f"Training on {len(X_train)} images, validating on {len(X_val)} images.")

model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, alpha=1e-4,
                      solver='adam', verbose=10, random_state=42,
                      learning_rate_init=.001,
                      early_stopping=True, validation_fraction=0.1)

print("Training the final, most robust Neural Network model...")
model.fit(X_train, y_train)
print("Training complete.")


validation_score = model.score(X_val, y_val)
print(f"\nFinal model accuracy on the validation set: {validation_score:.4f}")
joblib.dump(model, 'digit_recognizer_mnist.joblib')

print("\n✅ Final model trained and saved as 'digit_recognizer_mnist.joblib'")
print("This is our most powerful version yet. Please upload it to Hugging Face.")

