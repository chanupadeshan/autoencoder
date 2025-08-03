# Autoencoder

An autoencoder is a neural network that learns a compressed representation (encoding) of data, typically for dimensionality reduction or feature learning. It consists of two main parts: an **encoder** and a **decoder**.

## Architecture

*   **Encoder**: Compresses the input data into a low-dimensional latent space.
    -   `z = f(x)`
*   **Decoder**: Reconstructs the original data from the latent representation.
    -   `x' = g(z)`
*   **Latent Space (Bottleneck)**: The layer containing the compressed representation. Its smaller dimensionality forces the network to learn meaningful features.
*   **Loss Function**: Measures the reconstruction error between the original input `x` and the reconstructed output `x'`. Common choices are Mean Squared Error (MSE) or Binary Cross-entropy.

## Common Types

- **Vanilla**: The simplest form of an autoencoder.
- **Sparse**: Adds a sparsity penalty to the loss function to regularize the model.
- **Denoising**: Trained to reconstruct a clean input from a corrupted version.
- **Variational (VAE)**: A generative model that learns a probabilistic mapping to the latent space.
- **Convolutional**: Uses convolutional layers, making it suitable for image data.

## Applications

- Dimensionality Reduction
- Feature Learning
- Data Denoising
- Anomaly Detection
- Image Compression

## Training

The network is trained by minimizing the reconstruction loss, which forces the decoder to learn how to reconstruct the original data from the encoder's compressed representation.
