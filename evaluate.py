import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

MODEL_PATH = './pneumonia_model.h5' 
TEST_DIR = './chest_xray/test/'
IMAGE_SIZE = (150, 150) 
BATCH_SIZE = 32

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

print(f"Loading model from {MODEL_PATH}:")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

print(f"Loading test data from {TEST_DIR}:")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    class_names=CLASS_NAMES, 
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0 # Use 0-1 scaling
    return image, label

test_dataset = test_dataset.map(normalize_image)
print("Data loaded and normalized successfully.")

print("Making predictions on test data:")
predictions_raw = model.predict(test_dataset)

true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
print(f"Total predictions made: {len(predictions_raw)}")
print(f"Total true labels found: {len(true_labels)}")

predicted_labels = [1 if p > 0.5 else 0 for p in predictions_raw]

print("\n Model Evaluation Results:")

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))

print("Confusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

print("\n(Rows are True Labels, Columns are Predicted Labels)")
print(f"                 Predicted {CLASS_NAMES[0]} | Predicted {CLASS_NAMES[1]}")
print(f"True {CLASS_NAMES[0]:<12} | {cm[0][0]:<18} | {cm[0][1]}")
print(f"True {CLASS_NAMES[1]:<12} | {cm[1][0]:<18} | {cm[1][1]}")