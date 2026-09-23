import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_training_pipeline():
    """
    Loads MRI image data, builds a CNN model from scratch, trains it,
    and saves the model and class names.
    """
    print("--- Training Pipeline Started ---")

    print("1/4 - Loading and preparing image data...")
    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 16
    DATA_DIR = "brain_tumor_dataset"

    train_dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary' 
    )
    
    validation_dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    class_names = train_dataset.class_names
    
    print("2/4 - Building the CNN model...")
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy'],
    )
    model.summary()

    print("3/4 - Training the model (this may take a few minutes)...")
    epochs = 10
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )

    print("4/4 - Saving artifacts...")
    model.save('brain_tumor_model.keras')
    joblib.dump(class_names, 'class_names.pkl')
    joblib.dump(history.history, 'training_history.pkl')
    
    print("--- Training Pipeline Completed Successfully! ---")
if __name__ == "__main__":
    run_training_pipeline()