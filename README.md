# GoogLeNet (Inception v1) — PyTorch Implementation

A clean and modular implementation of **GoogLeNet (Inception v1)** in **PyTorch**, built from scratch.  
The model supports both **training** and **inference** modes with auxiliary classifiers as described in the original paper:

> **Going Deeper with Convolutions** — Szegedy et al., 2015

---

## Overview

This implementation follows the original GoogLeNet (Inception v1) architecture, consisting of:

- Initial convolution and pooling layers  
- 9 Inception modules  
- 2 Auxiliary classifiers (`InceptionAux`) used during training  
- Global Average Pooling followed by Dropout and a fully connected layer  

---

## Components

### Inception Module
Each Inception block consists of four parallel branches:

1. **1×1 Convolution**  
2. **1×1 → 3×3 Convolution**  
3. **1×1 → 5×5 Convolution**  
4. **3×3 MaxPool → 1×1 Convolution**

All outputs are concatenated along the channel dimension.

### Auxiliary Classifier
Auxiliary branches are used during training to improve gradient flow and regularization.  
Each consists of:
- Average Pooling  
- 1×1 Convolution  
- Fully Connected Layer  
- Dropout + Linear + Softmax

---
## Reference
Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions.
[Read the Paper](https://arxiv.org/abs/1409.4842)

---
## Author
**Muhammad Ali**
