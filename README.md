# Autoencoder

## What is an Autoencoder?

An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal "noise".

## Architecture

An autoencoder consists of two main parts:

### 1. Encoder

- Maps the input data to a lower-dimensional latent space representation
- Compresses the input into a compact encoded representation
- Formula: `z = f(x)` where `x` is input and `z` is the encoded representation

### 2. Decoder

- Reconstructs the original input from the encoded representation
- Maps from the latent space back to the original input space
- Formula: `x' = g(z)` where `z` is the encoded representation and `x'` is the reconstruction

## Mathematical Foundation

### Basic Autoencoder Formulation

An autoencoder consists of two functions:

**Encoder function:**
```math
z = f_θ(x) = σ(W_1x + b_1)
```

**Decoder function:**
```math
x' = g_φ(z) = σ(W_2z + b_2)
```

Where:
- `x` ∈ ℝ^d is the input vector
- `z` ∈ ℝ^k is the latent representation (k < d for compression)
- `x'` ∈ ℝ^d is the reconstructed output
- `θ = {W_1, b_1}` are encoder parameters
- `φ = {W_2, b_2}` are decoder parameters
- `σ` is the activation function

### Loss Functions

**Mean Squared Error (MSE):**
```math
L_{MSE}(x, x') = \frac{1}{n} \sum_{i=1}^{n} ||x_i - x'_i||^2
```

**Binary Cross-Entropy (for binary data):**
```math
L_{BCE}(x, x') = -\sum_{i=1}^{d} [x_i \log(x'_i) + (1-x_i) \log(1-x'_i)]
```

**Mean Absolute Error (MAE):**
```math
L_{MAE}(x, x') = \frac{1}{n} \sum_{i=1}^{n} |x_i - x'_i|
```

### Optimization Objective

The autoencoder learns by minimizing the reconstruction loss:

```math
\min_{θ,φ} \frac{1}{N} \sum_{i=1}^{N} L(x_i, g_φ(f_θ(x_i)))
```

Where N is the number of training samples.

### Regularization

**Sparse Autoencoder:**
```math
L_{sparse} = L_{reconstruction} + λ \sum_{j=1}^{k} KL(ρ || ρ_j)
```

Where:
- `ρ` is the target sparsity parameter
- `ρ_j` is the average activation of hidden unit j
- `KL` is the Kullback-Leibler divergence
- `λ` controls sparsity strength

**Denoising Autoencoder:**
```math
L_{denoise} = E_{x∼p_{data}} E_{x̃∼q(x̃|x)} [L(x, g_φ(f_θ(x̃)))]
```

Where `x̃` is the corrupted input and `q(x̃|x)` is the noise distribution.

## Key Components

### Latent Space (Bottleneck)

- The compressed representation of the input
- Forces the network to learn meaningful features
- Dimensionality is typically much smaller than input

### Loss Function

- Measures the difference between input and reconstruction
- Common choices: MSE, Binary Cross-entropy, MAE
- Goal: Minimize reconstruction error

## Types of Autoencoders

### 1. Vanilla Autoencoder
Basic encoder-decoder architecture with standard loss:
```math
L = ||x - g_φ(f_θ(x))||^2
```

### 2. Sparse Autoencoder
Adds sparsity constraint to encourage sparse representations:
```math
L_{total} = L_{reconstruction} + λ \sum_{j=1}^{h} KL(ρ || ρ_j)
```
Where `ρ_j = \frac{1}{m} \sum_{i=1}^{m} a_j^{(i)}` is the average activation.

### 3. Denoising Autoencoder
Trained to reconstruct clean data from corrupted input:
```math
L = E_{x∼p_{data}} E_{x̃∼q(x̃|x)} [||x - g_φ(f_θ(x̃))||^2]
```

### 4. Variational Autoencoder (VAE)
Probabilistic approach with regularized latent space:
```math
L_{VAE} = E_{q_φ(z|x)}[\log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```
Where the latent variables follow: `z ~ N(μ(x), σ²(x))`

### 5. Convolutional Autoencoder
Uses CNN layers for spatial data with convolution and deconvolution operations:
```math
z = f_{conv}(x) = σ(Conv(x) + b)
```
```math
x' = g_{deconv}(z) = σ(Deconv(z) + b')
```

## Applications

- **Dimensionality Reduction**: Compress high-dimensional data
- **Feature Learning**: Learn meaningful representations
- **Data Denoising**: Remove noise from corrupted data
- **Anomaly Detection**: Identify outliers based on reconstruction error
- **Image Compression**: Lossy compression of images
- **Data Generation**: Generate new samples (especially VAEs)

## Advantages

- Unsupervised learning (no labeled data required)
- Flexible architecture adaptable to different data types
- Can learn non-linear transformations
- Useful for preprocessing and feature extraction

## Limitations

- May lose important information during compression
- Training can be unstable
- Prone to overfitting with limited data
- Quality depends heavily on architecture design

## Training Process

### Forward Pass
```math
z = f_θ(x) = σ_1(W_1x + b_1)
```
```math
x' = g_φ(z) = σ_2(W_2z + b_2)
```

### Loss Calculation
```math
L = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - x'^{(i)}||^2
```

### Backward Pass (Gradient Computation)
```math
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial x'} \frac{\partial x'}{\partial W_2}
```
```math
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial x'} \frac{\partial x'}{\partial z} \frac{\partial z}{\partial W_1}
```

### Parameter Updates (Gradient Descent)
```math
W_1 := W_1 - α \frac{\partial L}{\partial W_1}
```
```math
W_2 := W_2 - α \frac{\partial L}{\partial W_2}
```

Where `α` is the learning rate.

### Training Steps:

1. Forward pass: Input → Encoder → Latent representation → Decoder → Reconstruction
2. Calculate reconstruction loss
3. Backpropagate error through the network
4. Update weights to minimize loss
5. Repeat until convergence