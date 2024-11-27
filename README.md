# Network Compression Using Singular Value Decomposition (SVD)

This work explores the use of **Singular Value Decomposition (SVD)** to compress neural networks, reducing their size while maintaining performance. The tasks involve approximating weight matrices with low-rank decompositions, fine-tuning compressed networks, and dynamically applying SVD during training. Below are the details of the three problems tackled in this project.

---

## Baseline Network Compression Using SVD

### **Objective**
Compress a fully-connected neural network trained on the MNIST dataset by approximating its weight matrices using SVD.

### **Process**
1. Apply **SVD** to the weight matrices \( W^{(l)} \) (except the final layer), approximating them as:
   \[
   W^{(l)} \approx \hat{W}^{(l)} = U^{(l)} S^{(l)} V^{(l)T}
   \]
   where \( U^{(l)} \), \( S^{(l)} \), and \( V^{(l)} \) are derived from the SVD of \( W^{(l)} \).
2. Using only the top \( D \) singular values from \( S^{(l)} \) to construct the low-rank approximation:
   \[
   W^{(l)} \approx U^{(l)}_{:,1:D} S^{(l)}_{1:D,1:D} V^{(l)T}_{:,1:D}
   \]
3. Evaluating the test accuracy for different values of \( D \) (e.g., 10, 20, 50, 100, etc.).

---

## Fine-Tuning Compressed Networks

### **Objective**
Improve the performance of networks compressed using SVD by fine-tuning their parameters.

### **Process**
1. Defining a new network where weight matrices are factorized into low-rank components:
   \[
   W^{(l)} = U^{(l)} V^{(l)T}
   \]
2. Modifying the feedforward pass to use these factorized weights:
   \[
   x^{(l+1)} = g\left(U^{(l)} V^{(l)T} x^{(l)} + b^{(l)}\right)
   \]
3. Initializing \( U^{(l)} \) and \( V^{(l)} \) using the top \( D = 20 \) components:
   \[
   U^{(l)} = U_{:,1:20}^{(l)}, \quad V^{(l)T} = S_{1:20,1:20}^{(l)} V_{:,1:20}^{(l)T}
   \]
4. Fine-tuning the network using backpropagation with a smaller learning rate.
   
---

## Dynamic SVD During Training

### **Objective**
Incorporate SVD into the training process by dynamically applying it at every iteration.

### **Steps**
1. Initializing weights \( W^{(l)} \) using the baseline model.
2. During each feedforward pass:
   - Performing SVD on \( W^{(l)} \):
     \[
     W^{(l)} = U_{:,1:20}^{(l)} S_{1:20,1:20}^{(l)} V_{:,1:20}^{T(l)}
     \]
   - Using only the top \( D = 20 \) components for feedforward computation:
     \[
     x^{(l+1)} = g\left(U_{:,1:20}^{(l)} S_{1:20,1:20}^{(l)} V_{:,1:20}^{T(l)} x^{(l)} + b^{(l)}\right)
     \]
3. Updating parameters \( W^{(l)} \) during backpropagation while ensuring:
   - Gradients are computed with respect to the full weight matrix.
   - The derivative of the approximation function is treated as identity (\(\frac{\partial f(W^{(l)})}{\partial W} = 1\)) for simplicity.

---

## Key Insights

- **Compression vs Accuracy Tradeoff**: Reducing \( D \) leads to smaller networks but may degrade accuracy.
- **Fine-Tuning**: Initializing compressed networks with pre-computed SVD results and fine-tuning improves performance.
- **Dynamic Compression**: Applying SVD dynamically during training can further boost accuracy while keeping memory usage low.

---

## Mathematical Summary

### Low-Rank Approximation
\[
W^{(l)} \approx U_{:,1:D}^{(l)} S_{1:D,1:D}^{(l)} V_{:,1:D}^{T(l)}
\]

### Feedforward Pass
- Compressed Network:
\[
x^{(l+1)} = g\left(U_{:,1:D}^{(l)} V_{:,1:D}^{T(l)} x^{(l)} + b^{(l)}\right)
\]

- Dynamic Compression:
\[
x^{(l+1)} = g\left(U_{:,1:D}^{(l)} S_{1:D,1:D}^{(l)} V_{:,1:D}^{T(l)} x^{(l)} + b^{(l)}\right)
\]

### Backpropagation
- Update Rule:
\[
W^{(l+1)} = W^{(l)} - \eta \frac{\partial f(W^{(l)})}{\partial W}
\]

Where:
- \( f(W) = W \), assuming identity derivative for simplicity.

---

## Results
- **Baseline Network** achieves high accuracy but is computationally expensive.
- **Compressed Networks** reduce memory usage significantly while maintaining competitive accuracy when fine-tuned.
- **Dynamic Compression** provides additional performance gains by incorporating compression-aware training.
