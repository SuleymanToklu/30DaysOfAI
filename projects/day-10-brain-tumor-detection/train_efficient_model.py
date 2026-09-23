import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_efficient_training_pipeline():
    """
    Builds an EFFICIENT model with GlobalAveragePooling2D to drastically
    reduce parameters and prevent overfitting. This is the way.
    """
    print("--- Efficient Model Training Pipeline Started ---")

    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 16
    DATA_DIR = "brain_tumor_dataset"

    train_dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
    )
    validation_dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
    )
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("--- Building the EFFICIENT model ---")
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    print("--- Training the EFFICIENT model ---")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=30, 
        callbacks=[early_stopping, reduce_lr]
    )

    model.save('brain_tumor_model_efficient.keras')
    joblib.dump(history.history, 'training_history_efficient.pkl')
    
    model.summary()
    print("--- Efficient Model Training Completed Successfully! ---")


if __name__ == "__main__":
    run_efficient_training_pipeline()