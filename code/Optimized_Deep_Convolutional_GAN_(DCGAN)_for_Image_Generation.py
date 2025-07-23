import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration and Hyperparameters ---
# You should tune these based on your specific dataset and computational resources.
# For demonstration, we'll use MNIST (grayscale 28x28 images).
# For higher resolution or color images (e.g., CIFAR-10, CelebA), adjust these.
IMAGE_SIZE = 28
CHANNELS = 1 # 1 for grayscale (MNIST), 3 for RGB (CIFAR-10, CelebA)
NOISE_DIM = 100 # Dimension of the random noise vector (latent space)
BATCH_SIZE = 64
EPOCHS = 100 # GANs often require many epochs to converge and produce good results
LEARNING_RATE_G = 0.0002 # Learning rate for the Generator
LEARNING_RATE_D = 0.0002 # Learning rate for the Discriminator
BETA1 = 0.5 # Beta1 hyperparameter for Adam optimizer (recommended for GANs)

# Directory to save generated images during training
OUTPUT_DIR = 'gan_generated_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Data Loading and Preprocessing ---
print("Loading MNIST dataset...")
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshape images to (batch_size, height, width, channels)
train_images = train_images.reshape(train_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, CHANNELS).astype('float32')

# Normalize images to [-1, 1] range (common for GANs with tanh output)
# MNIST pixel values are 0-255. (x - 127.5) / 127.5 scales them to [-1, 1].
train_images = (train_images - 127.5) / 127.5

# Create a tf.data.Dataset for efficient batching and shuffling
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(10000).batch(BATCH_SIZE)

print(f"Training data shape: {train_images.shape}")
print(f"Image value range: [{train_images.min()}, {train_images.max()}]")

# --- 2. Generator Model ---
# The Generator takes a random noise vector as input and outputs an image.
# It uses Conv2DTranspose (deconvolutional) layers to upsample the input.
# BatchNormalization stabilizes training.
# LeakyReLU is often used in the generator for better gradient flow.
# The final layer uses tanh activation to output pixel values in [-1, 1].

print("Building the Generator model...")
def make_generator_model():
    model = models.Sequential()

    # Input: Noise vector (NOISE_DIM,)
    # Reshape to a small spatial dimension (e.g., 7x7) with many channels
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256))) # Output shape: (None, 7, 7, 256)

    # Upsampling block 1: From 7x7 to 14x14
    # Conv2DTranspose (deconvolution) effectively upsamples
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) # Output shape: (None, 7, 7, 128)

    # Upsampling block 2: From 14x14 to 28x28
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) # Output shape: (None, 14, 14, 64)

    # Final output layer: From 28x28 to 28x28 with CHANNELS
    # Use tanh activation to match the [-1, 1] range of normalized images
    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # Output shape: (None, 28, 28, 1) for MNIST

    return model

generator = make_generator_model()
generator.summary()

# --- 3. Discriminator Model ---
# The Discriminator takes an image (real or fake) as input and outputs a single scalar
# representing the probability that the image is real.
# It uses Conv2D layers for downsampling.
# LeakyReLU is used for all layers except the output.
# Dropout is often added to the Discriminator to prevent it from overpowering the Generator too quickly.
# The final layer uses sigmoid activation for binary classification (real/fake).

print("Building the Discriminator model...")
def make_discriminator_model():
    model = models.Sequential()

    # Input: Image (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    # Downsampling block 1: From 28x28 to 14x14
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMAGE_SIZE, IMAGE_SIZE, CHANNELS]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # Dropout for stability

    # Downsampling block 2: From 14x14 to 7x7
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and output
    model.add(layers.Flatten())
    # Output: Single scalar probability (real or fake)
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

discriminator = make_discriminator_model()
discriminator.summary()

# --- 4. Define Loss Functions and Optimizers ---
# DCGAN uses Binary Crossentropy for both generator and discriminator losses.
# The Discriminator tries to classify real images as 1 and fake images as 0.
# The Generator tries to fool the Discriminator by making it classify fake images as 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) # from_logits=False because sigmoid is used

# Discriminator Loss: Sum of real loss and fake loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator Loss: How well the generator fools the discriminator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Separate optimizers for Generator and Discriminator (crucial for GANs)
# Adam optimizer with specific learning rates and beta1 (as recommended in DCGAN paper)
generator_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA1)
discriminator_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA1)

# --- 5. Training Step Function ---
# This function defines a single training step for both the generator and discriminator.
# It uses tf.function for performance optimization (compiles the function into a callable TensorFlow graph).

@tf.function
def train_step(images):
    # 1. Generate random noise for the generator input
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # Use tf.GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator's output for real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients to update model weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# --- 6. Image Generation for Visualization ---
# Helper function to generate and save images during training.
# This helps monitor the training progress and quality of generated samples.
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    # Rescale images from [-1, 1] to [0, 1] for plotting
    predictions = (predictions * 0.5) + 0.5

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Squeeze to remove the channel dimension if it's 1 (for grayscale)
        plt.imshow(predictions[i, :, :, 0] if CHANNELS == 1 else predictions[i, :, :, :])
        plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, f'image_at_epoch_{epoch:04d}.png'))
    plt.close() # Close the figure to free up memory

# Fixed noise for consistent visualization of generator progress
seed = tf.random.normal([16, NOISE_DIM]) # Generate 16 images for visualization

# --- 7. Full Training Loop ---
print("Starting GAN training...")
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        gen_losses = []
        disc_losses = []

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_losses.append(g_loss.numpy())
            disc_losses.append(d_loss.numpy())

        # Print progress
        avg_gen_loss = np.mean(gen_losses)
        avg_disc_loss = np.mean(disc_losses)
        print(f'Epoch {epoch + 1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time()-start:.2f} sec')

        # Generate and save images every few epochs
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

    # Generate final images after training
    generate_and_save_images(generator, epochs, seed)

# Run the training
train(train_dataset, EPOCHS)

# --- 8. Save Models ---
# Save the generator and discriminator models separately.
generator.save(os.path.join(OUTPUT_DIR, 'generator_model.h5'))
discriminator.save(os.path.join(OUTPUT_DIR, 'discriminator_model.h5'))
print(f"\nGenerator and Discriminator models saved to {OUTPUT_DIR}")

# --- Optional: Load and Generate New Images ---
# print("\nLoading saved generator and generating new images...")
# loaded_generator = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'generator_model.h5'))
# new_noise = tf.random.normal([10, NOISE_DIM])
# new_generated_images = loaded_generator(new_noise, training=False)
# new_generated_images = (new_generated_images * 0.5) + 0.5 # Rescale to [0, 1]

# fig = plt.figure(figsize=(2, 5))
# for i in range(new_generated_images.shape[0]):
#     plt.subplot(5, 2, i+1)
#     plt.imshow(new_generated_images[i, :, :, 0] if CHANNELS == 1 else new_generated_images[i, :, :, :])
#     plt.axis('off')
# plt.suptitle("Newly Generated Images")
# plt.show()
