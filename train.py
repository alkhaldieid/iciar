import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set up random seeds for reproducibility
seed = 1352
np.random.seed(seed)
tf.random.set_seed(seed)

# Image data directories and settings
train_dir = "./data/train"
validation_dir = "./data/validation"
test_dir = "./data/test"

img_size1 = 600
img_size2 = 800
crop_length = 600
BATCH_SIZE = 4


# Preprocessing function and image data generators
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


def create_gen(preprocessing_func):
    """Create an ImageDataGenerator with augmentation and preprocessing."""
    return ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
    )


def get_crops():
    """Generate training, validation, and test image crops."""
    data_gen = create_gen(tf.keras.applications.densenet.preprocess_input)

    train_batches = data_gen.flow_from_directory(
        train_dir,
        target_size=(img_size1, img_size2),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    validation_batches = data_gen.flow_from_directory(
        validation_dir,
        target_size=(img_size1, img_size2),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_batches = data_gen.flow_from_directory(
        test_dir,
        target_size=(img_size1, img_size2),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    train_crops = crop_generator(train_batches, crop_length=crop_length)
    val_crops = crop_generator(validation_batches, crop_length=crop_length)
    test_crops = crop_generator(test_batches, crop_length=crop_length)

    return train_crops, val_crops, test_crops


# Create and compile the model
def create_model():
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        pooling="avg",
        weights="imagenet",
        input_shape=(crop_length, crop_length, 3),
    )
    x = base_model.output
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers[:-6]:  # Fine-tune last few layers
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Define callbacks
checkpoint_path = "./model_checkpoint.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"
)
early_stopping = EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# Train the model
train_crops, val_crops, test_crops = get_crops()

model = create_model()

model.fit(
    train_crops,
    steps_per_epoch=376 // BATCH_SIZE,
    epochs=2000,
    validation_data=val_crops,
    validation_steps=12,
    callbacks=[checkpoint, reduce_lr, early_stopping],
)

# Evaluate the model
model.load_weights(checkpoint_path)
evaluation = model.evaluate(test_crops, steps=12)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Confusion matrix and classification report
test_batches = next(test_crops)  # Get a batch of test data
y_true = test_batches[1].argmax(axis=1)
y_pred = model.predict(test_batches[0]).argmax(axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
target_names = ["Benign", "InSitu", "Invasive", "Normal"]
print(classification_report(y_true, y_pred, target_names=target_names))
