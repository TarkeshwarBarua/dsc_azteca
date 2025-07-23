import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration and Hyperparameters ---
# You should tune these based on your specific dataset and computational resources.
IMG_HEIGHT = 32  # Adjust based on your image dimensions
IMG_WIDTH = 32   # Adjust based on your image dimensions
BATCH_SIZE = 64
EPOCHS = 50      # Start with a reasonable number, EarlyStopping will prevent overfitting
LEARNING_RATE = 0.001
NUM_CLASSES = 10 # Adjust based on your dataset's number of classes (e.g., 10 for CIFAR-10)

# Path to your dataset (if not using built-in datasets like CIFAR-10)
# TRAIN_DATA_DIR = 'path/to/your/train_data'
# VALIDATION_DATA_DIR = 'path/to/your/validation_data'

# --- 1. Data Loading and Preprocessing ---
# For demonstration, we'll use the CIFAR-10 dataset as it's a common image classification benchmark.
# If you have your own dataset, you'll need to load it using `ImageDataGenerator.flow_from_directory`
# or `tf.data.Dataset` pipelines.

print("Loading dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# --- 2. Data Augmentation for Robustness ---
# Data augmentation helps prevent overfitting by creating new training samples
# from existing ones by applying random transformations.
print("Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rotation_range=15,        # Randomly rotate images by 15 degrees
    width_shift_range=0.1,    # Randomly shift images horizontally by 10%
    height_shift_range=0.1,   # Randomly shift images vertically by 10%
    horizontal_flip=True,     # Randomly flip images horizontally
    zoom_range=0.1,           # Randomly zoom in on images by 10%
    fill_mode='nearest'       # Strategy for filling in new pixels created by transformations
)

# Fit the data generator to the training data
train_datagen.fit(x_train)

# No augmentation for test data, only normalization
test_datagen = ImageDataGenerator() # Only used for batching, no augmentation

# --- 3. Optimized CNN Model Architecture (VGG-like with Batch Normalization and Dropout) ---
# This architecture is designed for good performance on image classification tasks.
# Key optimization techniques used:
# - Convolutional layers for feature extraction.
# - Batch Normalization: Stabilizes and speeds up training by normalizing inputs to layers.
# - MaxPooling2D: Reduces spatial dimensions, helps with translation invariance.
# - Dropout: Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0.
# - ReLU activation: Non-linear activation function, computationally efficient.
# - Adam optimizer: Adaptive learning rate optimization algorithm, generally performs well.

print("Building the CNN model...")
def build_optimized_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25)) # Increased dropout for regularization

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3 (Optional, for deeper networks)
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten the feature maps for the fully connected layers
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5)) # Higher dropout for the dense layer

    # Output layer with softmax for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Build the model
model = build_optimized_cnn((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)

# --- 4. Model Compilation ---
# Compile the model with an optimizer, loss function, and metrics.
# Adam optimizer is chosen for its efficiency and good performance.
# Categorical Crossentropy is standard for multi-class classification with one-hot labels.
print("Compiling the model...")
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to see the architecture and number of parameters
model.summary()

# --- 5. Callbacks for Optimized Training ---
# Callbacks automate actions during training to improve efficiency and performance.
print("Setting up training callbacks...")
# Early Stopping: Stop training when a monitored metric (e.g., validation loss) stops improving.
# This prevents overfitting and saves training time.
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=10,        # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
)

# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
# This helps the model converge better in later stages of training.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Monitor validation loss
    factor=0.5,         # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=5,         # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.00001,     # Lower bound on the learning rate
    verbose=1           # Print messages when learning rate is reduced
)

callbacks = [early_stopping, reduce_lr]

# --- 6. Model Training ---
print("Starting model training...")
# Use `flow` method of ImageDataGenerator for training on batches of augmented data.
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_test, y_test), # Use the unaugmented test set for validation
    callbacks=callbacks,
    verbose=1
)

# --- 7. Model Evaluation ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 8. Plotting Training History ---
print("Plotting training history...")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# --- 9. Save the Trained Model ---
# It's good practice to save your trained model for future use.
model_save_path = "optimized_cnn_model.h5"
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

# --- Optional: Make Predictions ---
# print("\nMaking predictions on a sample from the test set...")
# sample_images = x_test[:5]
# sample_labels = y_test[:5]
# predictions = model.predict(sample_images)

# for i, pred in enumerate(predictions):
#     print(f"Image {i+1}:")
#     print(f"  True Label: {np.argmax(sample_labels[i])}")
#     print(f"  Predicted Label: {np.argmax(pred)}")
#     print(f"  Prediction Probabilities: {pred.round(2)}")
