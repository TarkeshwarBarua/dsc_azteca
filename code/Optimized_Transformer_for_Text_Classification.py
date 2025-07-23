import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import re # For text cleaning

# --- Configuration and Hyperparameters ---
# You should tune these based on your specific dataset and computational resources.
VOCAB_SIZE = 20000     # Max number of words to keep based on frequency
MAX_SEQUENCE_LENGTH = 256 # Max length of a sequence (e.g., number of words in a review)
EMBEDDING_DIM = 256    # Dimension of the word embeddings
NUM_HEADS = 8          # Number of attention heads in Multi-Head Attention
FF_DIM = 512           # Dimension of the Feed-Forward network in Transformer Block
NUM_TRANSFORMER_BLOCKS = 4 # Number of stacked Transformer encoder blocks
BATCH_SIZE = 32
EPOCHS = 30            # Start with a reasonable number, EarlyStopping will prevent overfitting
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1     # Dropout rate for regularization
NUM_CLASSES = 1        # 1 for binary classification (sentiment), adjust for multi-class

# --- 1. Data Loading and Preprocessing ---
# For demonstration, we'll use the IMDB movie review dataset for sentiment analysis (binary classification).
# This dataset consists of 50,000 movie reviews, labeled as positive or negative.
# If you have your own text dataset, you'll need to load it and prepare it similarly.

print("Loading IMDB dataset...")
# num_words limits the vocabulary size to the most frequent VOCAB_SIZE words
# The IMDB dataset is already pre-tokenized into integer sequences.
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

print(f"Original training data samples: {len(x_train)}")
print(f"Original test data samples: {len(x_test)}")

# Pad sequences to a fixed length
# 'post' padding means zeros are added at the end of the sequence
# 'post' truncating means sequences longer than MAX_SEQUENCE_LENGTH are cut from the end
x_train_padded = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Convert labels to float32 for binary crossentropy
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print(f"Padded training data shape: {x_train_padded.shape}")
print(f"Padded test data shape: {x_test_padded.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Create TensorFlow Datasets for efficient batching and caching
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_padded, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_padded, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 2. Custom Transformer Components ---

# Positional Encoding Layer: Adds positional information to word embeddings
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

# Multi-Head Self-Attention Layer: The core of the Transformer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.proj_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim) # Output projection

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True) # (..., seq_len, seq_len)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key) # Scale by sqrt(d_k)
        weights = tf.nn.softmax(scaled_score, axis=-1) # Attention weights
        output = tf.matmul(weights, value) # Weighted sum of values
        return output, weights

    def separate_heads(self, x, batch_size):
        # Reshape (batch_size, seq_len, embed_dim) to (batch_size, num_heads, seq_len, proj_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # Transpose for attention calculation

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)      # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, proj_dim)
        key = self.separate_heads(key, batch_size)      # (batch_size, num_heads, seq_len, proj_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, proj_dim)

        attention, weights = self.attention(query, key, value)
        # Recombine heads: (batch_size, seq_len, num_heads, proj_dim)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        output = self.combine_heads(concat_attention) # Final linear projection
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

# Transformer Block: Combines Multi-Head Attention and Feed-Forward Network
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Layer Normalization and Multi-Head Attention with Residual Connection
        attn_output = self.att(self.layernorm1(inputs))
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output # Residual connection

        # Layer Normalization and Feed-Forward Network with Residual Connection
        ffn_output = self.ffn(self.layernorm2(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# --- 3. Transformer Model Architecture ---
print("Building the Transformer model...")

def build_transformer_model(vocab_size, max_sequence_length, embed_dim,
                            num_heads, ff_dim, num_transformer_blocks,
                            dropout_rate, num_classes):
    inputs = layers.Input(shape=(max_sequence_length,))
    x = PositionalEmbedding(max_sequence_length, vocab_size, embed_dim)(inputs)

    # Stack multiple Transformer blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Global Average Pooling to get a fixed-size representation
    # This is common for classification tasks after the encoder blocks
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x) # Dropout before the final dense layer
    x = layers.Dense(20, activation="relu")(x) # Optional: small dense layer before output
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x) # Binary classification
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x) # Multi-class classification

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build the model
model = build_transformer_model(VOCAB_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
                                NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS,
                                DROPOUT_RATE, NUM_CLASSES)

# --- 4. Model Compilation ---
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
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# --- 7. Model Evaluation ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(test_dataset, verbose=0)
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
# Custom layers require special handling for saving/loading.
model_save_path = "optimized_transformer_model.h5"
model.save(model_save_path, save_format='h5') # Use HDF5 format for custom layers
print(f"\nModel saved to {model_save_path}")

# --- Optional: Make Predictions ---
# print("\nMaking predictions on a sample from the test set...")
# # To make predictions on new raw text, you'd need the original tokenizer
# # used during training. For IMDB, we can simulate it.
# word_index = keras.datasets.imdb.get_word_index()
# # Add special tokens for padding, start, unknown, unused
# # These are typically added by Keras's Tokenizer or dataset loaders
# word_index = {k:(v+3) for k,v in word_index.items()}
# word_index["<PAD>"] = 0
# word_index["<START>"] = 1
# word_index["<UNK>"] = 2
# word_index["<UNUSED>"] = 3

# def preprocess_text_for_prediction(text, tokenizer_word_index, max_len):
#     # Basic cleaning (lowercase, remove punctuation)
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     words = text.split()
#     # Convert words to indices, use <UNK> for unknown words
#     encoded = [tokenizer_word_index.get(word, tokenizer_word_index["<UNK>"]) for word in words]
#     # Pad sequence
#     padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')
#     return padded

# sample_texts = [
#     "This movie was absolutely fantastic! A must-watch.",
#     "I hated every minute of it. What a waste of time.",
#     "It was okay, not great, not terrible.",
#     "The acting was superb, but the plot was a bit weak."
# ]

# # Load the model with custom objects
# loaded_model = keras.models.load_model(
#     model_save_path,
#     custom_objects={
#         "PositionalEmbedding": PositionalEmbedding,
#         "MultiHeadSelfAttention": MultiHeadSelfAttention,
#         "TransformerBlock": TransformerBlock
#     }
# )

# for text in sample_texts:
#     processed_input = preprocess_text_for_prediction(text, word_index, MAX_SEQUENCE_LENGTH)
#     prediction = loaded_model.predict(processed_input)[0][0]
#     sentiment = "Positive" if prediction > 0.5 else "Negative"
#     print(f"Review: '{text}'")
#     print(f"  Predicted Sentiment: {sentiment} (Probability: {prediction:.4f})\n")
