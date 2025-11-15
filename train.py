import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
TRAIN_DIR = './chest_xray/train/'
TEST_DIR = './chest_xray/test/'
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='binary', image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=True
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels='inferred', label_mode='binary', image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=False
)
rescaling_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (rescaling_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (rescaling_layer(x), y))
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(factor=0.1),
])
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ModelCheckpoint to save the best model
checkpoint_callback = ModelCheckpoint(
    filepath="pneumonia_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# EarlyStopping to prevent wasting time
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\nStarting training with Checkpoint and EarlyStopping...")

history = model.fit(
    train_dataset,
    epochs=25, 
    validation_data=test_dataset,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

print("\nTraining complete.")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()