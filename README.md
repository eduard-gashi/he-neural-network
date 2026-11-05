# he-neural-network

Privacy-preserving neural network in C++ using OpenFHE.

This project explores encrypted training and inference and analyzes their impact on runtime, memory usage, and practical feasibility.

## Dependencies

The project depends on:

- **OpenFHE** for homomorphic encryption
- **Eigen** for matrix and vector operations

### Eigen

Eigen is a **header-only** C++ template library. This means no separate binary library has to be linked. Only the Eigen header files are required, and the compiler must be able to find the corresponding include directory.

Typical usage in the source code:

```cpp
#include <Eigen/Dense>
```

To make this work, add the directory containing the `Eigen/` folder to your compiler include path.


## Data

The dataset used in this project consists of a 5x2 input matrix and binary labels:

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

The model is a simple single-neuron network with a sigmoid activation function and mean squared error (MSE) loss.

## NeuralNetworkCipher

Homomorphically encrypted training is implemented in `NeuralNetworkCipher.cpp`.  
Matrix multiplications are approximated using ciphertext rotations and masking operations.
The sigmoid activation function is approximated using the polynomial Chebychev approximation with a polynom degree of 3.


### Encrypted Matrix Multiplication

The most expensive part and technically difficult part of encrypted neural-network training is matrix multiplication.

The function `Encoder::matmulXtDelta` computes the encrypted matrix-vector product:

\[
X^t * delta
\]

where \(X^t\) contains the transposed input features and \(\delta\) contains the encrypted error terms used for the gradient update.

The computation is performed entirely on ciphertexts:

1. **Slot-wise multiplication**  
   `EvalMult(Xt, delta)` multiplies the packed values of `Xt` and `delta` component-wise. This produces the individual products \(X_{ij}\delta_i\) across the ciphertext slots.

2. **Block-wise summation**  
   `sumSlots(ct_prod, num_features)` sums the products belonging to one feature. This reduces each feature block to a single accumulated value in slot 0.

3. **Rotation for the next feature**  
   `EvalRotate(Xt, 5)` rotates the packed feature matrix so that the values of the next feature are aligned with `delta`. The multiplication and block-wise summation are repeated to compute the second encrypted dot product.

4. **Merging partial results**  
   `EvalMerge(results)` combines the accumulated values from both feature computations into a single ciphertext.

5. **Result alignment and replication**  
   `EvalRotate(ct_sum_matrix, -1)` places the resulting gradient values into the correct slot positions. `_repeatBlock(ct_sum_matrix, 15, 3)` then replicates the result so that it can be used efficiently in the subsequent encrypted update step.

No intermediate result is decrypted. However, the approach requires ciphertext rotations, which need additional rotation keys and contribute significantly to runtime and multiplicative depth.


## Cryptographic Hyperparameters

Several cryptographic hyperparameters influence encrypted training, especially runtime, memory consumption, and numerical precision. The hyperparameters can be changed in config.h.

### security_level

The security level is one of the most important parameters. It determines whether the scheme provides, for example, 128-bit or 256-bit security. Higher security requirements generally imply a larger ring dimension and a longer modulus chain in the CKKS scheme.

The ring dimension defines the size of the underlying polynomial ring used for ciphertexts. As a result, even very small plaintext datasets can expand into much larger ciphertext objects in memory.


### mult_depth

mult_depth specifies the number of sequential multiplications that can be performed on a ciphertext before the accumulated approximation error becomes too large.

In this implementation, each training epoch requires a multiplicative depth of 16. A higher multiplicative depth increases runtime and parameter sizes. Consequently, a larger multiplicative depth requires a greater ring dimension, which in turn increases storage requirements. For more complex neural networks or a larger number of epochs, bootstrapping is necessary to refresh the ciphertext and reset the noise budget.


### batch_size

`batch_size` determines how many values can be packed into a single ciphertext. In CKKS, one ciphertext can store up to `ring_dimension / 2` slots.

### scale_mod_size

`scale_mod_size` controls the precision with which floating-point values are represented inside the ciphertext. Larger values usually improve numerical precision, but they make encrypted computation slower shich makes training longer.

## Benchmarks

### Storage

**Cleartext training data**  
The cleartext training data consists of 15 double values. Since one `double` occupies 8 bytes, the total memory footprint is:

15 × 8B = 120B

If the size of the MatrixXd-Object is added, storage becomes:

120B + 24B = 144B

**Ciphertext training data**  
The size of a ciphertext object depends on the chosen security level, multiplicative depth, and scale modulus size. These parameters determine the required ring dimension, which determines the storage size of a ciphertext.

For one epoch and a security level of 128-Bit, a multiplicative depth of 16 and a ring dimension of `2^15 = 32.768` were required, resulting in a ciphertext size of `9.437.184 bytes`.

## Ciphertext training data

The storage footprint increased from 144 B to 9 MB. However, a single CKKS ciphertext can pack up to Ringdimension/2 slots, so with a ring dimension of `2^15 = 32.768`, we can encode `2^14 = 16.384` double values in one ciphertext.

Since each double requires 8B, this corresponds to:


`16.384 x 8B = 131.072 = 0.125MB`

Including an additional 24 B of Eigen-Object size gives:

`131.072 + 24B = 131.096 = 0.125MB`

Therefore, the ciphertext storage expansion factor is:

`9MB \ 0.125MB = 72`

### Runtime

The main program was executed 30 times on a Intel Core Ultra 7 155H (3.80 GHz). The measured results were as follows:

Cleartext:
Mean = `0.14984ms`, Standard Deviation = `0.03171ms`

Ciphertext:
Mean = `2175.76ms`, Standard Deviation = `339.97ms`

The training duration increased by a factor of 14,520, demonstrating the impracticality of training encrypted complex neural networks on real-world-sized datasets. Encrypted inference however is way more practical, since it requires way less computations than training.


### Precision

The numerical precision is primarily determined by `scale_mod_size`. For larger numbers of epochs, a higher `scale_mod_size` is required to preserve comparable numerical accuracy.

Lower values of `scale_mod_size` improve performance but reduce precision.


## How to Run

### Prerequisites

Make sure the following are installed:

- A C++ compiler with C++17 support
- CMake
- Ninja
- OpenFHE
- Eigen

### Build

From the project root, create and enter a build directory:

```bash
mkdir build
cd build
```

Configure the project with CMake and generate Ninja build files:

```bash
cmake .. -G Ninja
```

Build the project using Ninja. The `-j` option specifies the number of parallel build jobs. It can be replaced by the number of CPU cores you want to use:

```bash
ninja -j4
```

Alternatively, use all available CPU cores:

```bash
ninja -j$(nproc)
```

On Windows PowerShell, use:

```powershell
ninja -j$env:NUMBER_OF_PROCESSORS
```

### Run

After a successful build, run the executable from the `build` directory:

```bash
./main.exe
```

On Windows, you can also run:

```powershell
.\main.exe
```

### Clean Rebuild

To remove the build files and start again:

```bash
cd ..
rm -rf build
mkdir build
cd build
cmake .. -G Ninja
ninja -j4
```