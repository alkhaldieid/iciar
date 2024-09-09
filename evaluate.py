import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data directories and settings
test_dir = "./data/test"
img_size1 = 600
img_size2 = 800
crop_length = 600
BATCH_SIZE = 4


# Preprocessing function and image data generator for testing
def random_crop(img, random_crop_size):
    """Randomly crop an image to a specific size."""
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length=299):
    """Generate random crops for each batch of images."""
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


def create_test_gen(preprocessing_func):
    """Create an ImageDataGenerator for testing."""
    data_generator = ImageDataGenerator(preprocessing_function=preprocessing_func)

    test_batches = data_generator.flow_from_directory(
        test_dir,
        target_size=(img_size1, img_size2),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    return crop_generator(test_batches, crop_length=crop_length)


# Load the model
model_path = "./model_checkpoint.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Trained model weights not found. Please train the model first."
    )

model = tf.keras.models.load_model(model_path)

# Generate test crops
test_crops = create_test_gen(tf.keras.applications.densenet.preprocess_input)

# Evaluate the model
print("Evaluating the model...")
evaluation = model.evaluate(test_crops, steps=12)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Get the confusion matrix and classification report
print("Generating confusion matrix and classification report...")

test_batches = next(test_crops)  # Get a batch of test data
y_true = test_batches[1].argmax(axis=1)
y_pred = model.predict(test_batches[0]).argmax(axis=1)

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
target_names = ["Benign", "InSitu", "Invasive", "Normal"]
print(classification_report(y_true, y_pred, target_names=target_names))
