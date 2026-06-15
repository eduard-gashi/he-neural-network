# he-neural-network

Privacy-preserving neural network in C++ using OpenFHE.

This project explores encrypted training and inference and analyzes their impact on runtime, memory usage, and practical feasibility.

## Data

The dataset used in this project is small and simple. It consists of a 5x2 input matrix and binary labels:

```text
X =
[
  [0, 0],
  [3, 1],
  [0, 1],
  [3, 0],
  [2, 2],
]

y =
[
  0, 
  1, 
  0, 
  1, 
  1
]
```

## Network

The model is a simple single-neuron network with a ReLU activation function and mean squared error (MSE) loss.

## NeuralNetworkCipher

Homomorphically encrypted training is implemented in `NeuralNetworkCipher.cpp`.  
Matrix multiplications are approximated using ciphertext rotations and masking operations.
The relu activation function is approximated using the polynomial Chebychev approximation.

## Cryptographic Hyperparameters

Several cryptographic hyperparameters influence encrypted training, especially runtime, memory consumption, and numerical precision.

### security_level

The security level is one of the most important parameters. It affects the ring dimension and the modulus chain used by the CKKS scheme.

The ring dimension determines the size of the underlying polynomials to which ciphertexts belong. As a result, even very small plaintext datasets can become large ciphertext objects in memory.

### mult_depth

`mult_depth` specifies how many sequential multiplications can be performed before the accumulated approximation error becomes too large.

In this implementation, each training epoch requires a multiplicative depth of 13. Higher multiplicative depth generally increases runtime and parameter sizes. For more complex neural networks or a larger number of epochs, bootstrapping would be required to refresh the ciphertext and reset the noise budget.

### batch_size

`batch_size` determines how many values can be packed into a single ciphertext.

In CKKS, one ciphertext can typically store up to `ring_dimension / 2` slots.

### scale_mod_size

`scale_mod_size` controls the precision with which floating-point values are represented inside the ciphertext.

Larger values usually improve numerical precision, but they make encrypted computation slower shich makes training longer.

## Benchmark

### Storage

**Cleartext training data**  
The cleartext training data consists of 15 double values. Since one `double` occupies 8 bytes, the total memory footprint is:

15 × 8 = 120 bytes

**Ciphertext training data**  
The size of a ciphertext object mainly depends on the chosen security level, multiplicative depth, and scale modulus size. These parameters determine the required ring dimension, which has a strong impact on memory consumption.

For one epoch, a multiplicative depth of 13 and a ring dimension of `2^15 = 32768` were required, resulting in a ciphertext size of `7,864,320 bytes`.

For two epochs, the ring dimension increased to `2^16 = 65536`, and the ciphertext size rose to `29,360,128 bytes`.

For three epochs, the ring dimension reached `2^17 = 131072`, producing a ciphertext size of `79,691,776 bytes`.

### Runtime

The main program was executed 30 times on a dedicated Linux server. The average runtime was measured to evaluate the computational overhead introduced by homomorphic encryption.

### Precision

The numerical precision is primarily determined by `scale_mod_size`. For larger numbers of epochs, a higher `scale_mod_size` is required to preserve comparable numerical accuracy.

Lower values of `scale_mod_size` improve performance but reduce precision.

For three epochs, the optimal `scale_mod_size` for precision was `33`, while the minimum usable value was `26`.