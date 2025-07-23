import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration and Hyperparameters ---
# You should tune these based on your specific dataset and computational resources.
VOCAB_SIZE = 10000     # Max number of words to keep based on frequency
MAX_SEQUENCE_LENGTH = 250 # Max length of a sequence (e.g., number of words in a review)
EMBEDDING_DIM = 128    # Dimension of the word embeddings
LSTM_UNITS = 64        # Number of units in the LSTM layer
BATCH_SIZE = 64
EPOCHS = 20            # Start with a reasonable number, EarlyStopping will prevent overfitting
LEARNING_RATE = 0.001
NUM_CLASSES = 1        # 1 for binary classification (sentiment), adjust for multi-class

# --- 1. Data Loading and Preprocessing ---
# For demonstration, we'll use the IMDB movie review dataset for sentiment analysis (binary classification).
# This dataset consists of 50,000 movie reviews, labeled as positive or negative.
# If you have your own text dataset, you'll need to load it and prepare it similarly.

print("Loading IMDB dataset...")
# num_words limits the vocabulary size to the most frequent VOCAB_SIZE words
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

print(f"Original training data samples: {len(x_train)}")
print(f"Original test data samples: {len(x_test)}")

# --- 2. Tokenization and Padding ---
# Text data needs to be converted into numerical sequences.
# Tokenizer converts words to integers (word indices).
# pad_sequences ensures all sequences have the same length, which is required for RNN input.

print("Tokenizing and padding sequences...")

# Since IMDB data is already tokenized into integer sequences, we just need padding.
# If you have raw text, you'd use:
# tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
# tokenizer.fit_on_texts(raw_train_texts)
# x_train_sequences = tokenizer.texts_to_sequences(raw_train_texts)
# x_test_sequences = tokenizer.texts_to_sequences(raw_test_texts)

# Pad sequences to a fixed length
# 'post' padding means zeros are added at the end of the sequence
x_train_padded = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Convert labels to float32 for binary crossentropy
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print(f"Padded training data shape: {x_train_padded.shape}")
print(f"Padded test data shape: {x_test_padded.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# --- 3. Optimized RNN Model Architecture (Bidirectional LSTM) ---
# This architecture is designed for good performance on sequential data tasks.
# Key optimization techniques used:
# - Embedding Layer: Converts positive integer indices (words) into dense vectors of fixed size.
# - Bidirectional LSTM: Processes the sequence in both forward and backward directions,
#   allowing the model to capture context from both past and future words in a sentence.
#   This is crucial for understanding nuances in text.
# - Dropout: Regularization to prevent overfitting.
# - Adam optimizer: Adaptive learning rate optimization algorithm.

print("Building the RNN model...")
def build_optimized_rnn(vocab_size, embedding_dim, max_sequence_length, lstm_units, num_classes):
    model = models.Sequential()

    # Embedding layer: Maps word indices to dense vectors
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=max_sequence_length))

    # Bidirectional LSTM layer: Processes sequence in both directions
    # return_sequences=True if you want to stack another recurrent layer
    model.add(layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)))
    model.add(layers.Dropout(0.5)) # Dropout after LSTM to prevent overfitting

    # Output layer
    # For binary classification (0 or 1), use sigmoid activation and 1 unit.
    # For multi-class classification, use 'softmax' activation and NUM_CLASSES units.
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Build the model
model = build_optimized_rnn(VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_UNITS, NUM_CLASSES)

# --- 4. Model Compilation ---
# Compile the model with an optimizer, loss function, and metrics.
# Adam optimizer is chosen for its efficiency and good performance.
# Binary Crossentropy is standard for binary classification problems.
print("Compiling the model...")
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

if NUM_CLASSES == 1:
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
else:
    # For multi-class, if labels are one-hot encoded: 'categorical_crossentropy'
    # If labels are integer encoded: 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', # or 'sparse_categorical_crossentropy'
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
    patience=5,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
)

# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
# This helps the model converge better in later stages of training.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Monitor validation loss
    factor=0.5,         # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=3,         # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.00001,     # Lower bound on the learning rate
    verbose=1           # Print messages when learning rate is reduced
)

callbacks = [early_stopping, reduce_lr]

# --- 6. Model Training ---
print("Starting model training...")
history = model.fit(
    x_train_padded, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_padded, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 7. Model Evaluation ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(x_test_padded, y_test, verbose=0)
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
model_save_path = "optimized_rnn_model.h5"
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

# --- Optional: Make Predictions ---
# print("\nMaking predictions on a sample from the test set...")
# sample_texts = [
#     "This movie was absolutely fantastic! A must-watch.",
#     "I hated every minute of it. What a waste of time.",
#     "It was okay, not great, not terrible.",
#     "The acting was superb, but the plot was a bit weak."
# ]

# # For prediction, you need to re-create a tokenizer if you didn't save it,
# # or load the original tokenizer used for training.
# # For IMDB, we can simulate the tokenization process.
# word_index = tf.keras.datasets.imdb.get_word_index()
# reverse_word_index = dict([(value + 3, key) for (key, value) in word_index.items()])
# # 0 is for padding, 1 for start of sequence, 2 for unknown
# reverse_word_index[0] = "<PAD>"
# reverse_word_index[1] = "<START>"
# reverse_word_index[2] = "<UNK>"
# reverse_word_index[3] = "<UNUSED>"

# def encode_text(text):
#     encoded = [word_index.get(word.lower(), 2) for word in text.split()]
#     return encoded

# sample_sequences = [encode_text(text) for text in sample_texts]
# sample_padded = pad_sequences(sample_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# predictions = model.predict(sample_padded)

# for i, pred in enumerate(predictions):
#     sentiment = "Positive" if pred[0] > 0.5 else "Negative"
#     print(f"Review: '{sample_texts[i]}'")
#     print(f"  Predicted Sentiment: {sentiment} (Probability: {pred[0]:.4f})\n")
