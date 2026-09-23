import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os
from tqdm import tqdm
import joblib

def load_data(data_dir):
    path = os.path.join(data_dir, 'labels_trainval.csv')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['image_path'] = df['frame'].apply(lambda x: os.path.join(data_dir, 'images', x))
    return df

def image_preprocessing_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (224, 224)) 
    return img

def create_classification_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_training_pipeline():
    print("--- Training Pipeline Started ---")
    DATA_DIR = "."
    
    print("1/5 - Loading data paths...")
    data = load_data(DATA_DIR)
    data = data.sample(n=15000, random_state=42)
    
    print("2/5 - Preprocessing images...")
    X_images = []
    y_labels = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Images"):
        img = image_preprocessing_pipeline(row['image_path'])
        if img is not None:
            X_images.append(img)
            y_labels.append(row['class_id'])

    X = np.array(X_images)
    y_labels = np.array(y_labels)

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("3/5 - Building and training the model...")
    num_classes = len(le.classes_)
    model = create_classification_model(num_classes)
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=32)

    print("4/5 - Saving the model...")
    model.save('road_object_model.keras')
    
    print("5/5 - Saving the class encoder...")
    joblib.dump(le, 'label_encoder.pkl') 
    
    print("--- Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_training_pipeline()