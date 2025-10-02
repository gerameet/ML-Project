# ML-Project

## Overview

This repository contains implementations and reports for three major assignments covering fundamental machine learning concepts, from basic regression to advanced deep learning architectures.

---

## Assignment 1: Classical ML Methods

### 1. Linear Regression & Optimization
- **Implementation:** Gradient Descent variants (Batch, Mini-batch, Stochastic)
- **Regularization:** Ridge (L2) and Lasso (L1) with hyperparameter tuning
- **Dataset:** Bangalore Food Delivery Time Prediction
- **Best Results:**
  - Stochastic GD: MSE = 149, R² = 0.65
  - Lasso (λ=0.25): MSE = 128, R² = 0.70
  - Ridge (λ=0.45): MSE = 150, R² = 0.65

**Key Insights:**
- Distance and weather are primary factors affecting delivery time
- Courier experience has minimal impact
- Feature scaling crucial for regularized models

### 2. K-Nearest Neighbors (KNN)
- **Tasks:** Classification and Retrieval (text-to-image, image-to-image)
- **Metrics:** Euclidean and Cosine distance
- **Performance:**
  - Classification: 91.94% accuracy (k=10)
  - Text-to-Image: MRR=1.0, Precision=0.974
  - Image-to-Image: MRR=0.935, Precision=0.841

### 3. Approximate Nearest Neighbors
**Locality Sensitive Hashing (LSH):**
- MRR: 0.601, Precision: 0.302
- Average comparisons: 226 (vs 50k for exact KNN)
- 99.5% hit rate with 222× speedup

**Inverted File Index (IVF):**
- nprobe=7: MRR=0.237, comparisons=7132
- Trade-off between accuracy and computational efficiency

### 4. Decision Trees for Fraud Detection
- **Splitting Criteria:** Entropy and Gini impurity
- **Best Configuration:** max_depth=12, min_samples_split=10
- **Performance:** 89.54% validation accuracy
- **Findings:**
  - Both criteria perform similarly
  - Depths >7 lead to overfitting
  - Time difference between transactions is key feature

### 5. Image Segmentation with SLIC
- **Algorithm:** Simple Linear Iterative Clustering
- **Color Spaces:** RGB vs LAB comparison
- **Optimization:** 3× speedup achieved (3s vs 9s per frame)
- **Parameters:** Number of clusters and compactness factor analysis

---

## Assignment 2: Neural Networks & Deep Learning

### 1. Multi-Layer Perceptron (MLP)

#### Multi-Class Classification
- **Architecture:** [64, 32] hidden layers
- **Best Config:** ReLU activation, Mini-batch GD, lr=0.01, batch_size=32
- **Performance:** 59.59% ± 2.99% accuracy (5-fold CV)
- **Key Learnings:** Lower std dev indicates better generalization

#### House Price Prediction (Bangalore)
- **Features:** 193 input features (total_sqft, bath, balcony, bhk)
- **Best Model:** 
  - Architecture: [193, 64, 32, 1]
  - Activation: tanh, Optimizer: minibatch
  - Test R²: 0.7716, RMSE: 58.26
- **Insights:** Mini-batch outperforms batch GD significantly

#### Multi-Label News Classification
- **Architecture:** [5000, 256, 128, 90]
- **Best Config:** ReLU + Momentum, lr=0.01
- **Performance:** F1-Micro: 0.016, Hamming Loss: 0.0138
- **Total experiments:** 48 configurations tested

### 2. Gaussian Mixture Models (GMM)
- **Application:** fMRI brain scan segmentation
- **Task:** Tissue type classification (3 labels)
- **Accuracy:** 81.74% pointwise accuracy
- **Challenges:** 
  - Boundary misclassification between tissue types
  - Lacks spatial context modeling
  - Overlap in feature distributions at transitions

### 3. Principal Component Analysis (PCA)
- **Dataset:** MNIST digit recognition
- **Variance Analysis:** First 150 components capture majority of variance
- **Reconstruction Quality:**
  - 500 PCs: Near-lossless
  - 150 PCs: Good quality, slight blur
  - 30 PCs: Significant detail loss

**MLP Classification on PCA-reduced features:**
- 500 dims: 97.17% accuracy
- 300 dims: 97.20% accuracy
- 150 dims: 97.45% accuracy
- 30 dims: 97.69% accuracy

**Insights:**
- Lower dimensions reduce computation while maintaining accuracy
- Maximum variance ≠ always most informative for classification
- PCA ineffective with low SNR or non-linear relationships

### 4. Autoencoders (AE)
- **Task:** Anomaly detection in MNIST
- **Method:** Reconstruction error (MSE) threshold
- **Bottleneck Comparison:**
  - Bottleneck=16: Highest AUC score
  - Bottleneck=8: Close performance
  - Bottleneck=2: Comparatively poor
- **Evaluation:** Precision, Recall, F1-score on anomaly detection

### 5. Variational Autoencoders (VAE)
- **Latent Space:** 2D visualization and analysis
- **Loss Functions Tested:**
  1. **KL + Reconstruction:** Smooth, continuous latent space (Loss: 151.46)
  2. **Reconstruction only:** Irregular clusters, poor interpolation (Loss: 147.95)
  3. **KL only:** Collapsed distribution, nonsensical outputs (Loss: 2.14e-7)

**Key Findings:**
- KL divergence essential for smooth latent space
- Reconstruction loss necessary for valid data mapping
- BCE outperforms MSE for binary images (sharper reconstructions)
- Successfully achieved meaningful 2D latent representations

---

## Assignment 3: Deep Learning with CNNs

### Age Prediction from Facial Images

**Dataset:** UTKFace (aligned and cropped facial images)

#### Custom CNN Architecture
- **Architecture:** 3 Convolutional Blocks
  - Block 1: 32 filters → MaxPool → Dropout(0.25)
  - Block 2: 64 filters → MaxPool → Dropout(0.25)
  - Block 3: 128 filters → MaxPool → Dropout(0.25)
  - Fully Connected: 512 units → Dropout(0.5) → 1 output
- **Features:** BatchNorm, ReLU activation, Data Augmentation
- **Training:** 12 epochs, Adam optimizer, ReduceLROnPlateau scheduler
- **Performance:**
  - Test MSE: 65.16
  - Mean Absolute Error: 5.91 years

#### ResNet-18 (Transfer Learning)
- **Approach:** Pretrained on ImageNet, fine-tuned for age regression
- **Modifications:** Replaced final FC layer for single output
- **Training:** 12 epochs, Adam optimizer with learning rate scheduling
- **Performance:**
  - Test MSE: 49.42
  - Mean Absolute Error: 4.88 years

**Why ResNet-18 Outperformed Custom CNN:**
1. **Residual Connections:** Skip connections prevent vanishing gradients in deeper networks
2. **Transfer Learning:** Pretrained weights capture low-level features (edges, textures, patterns)
3. **Architectural Advantages:** Bottleneck layers and deep feature hierarchies for better facial structure extraction
4. **Robustness:** Better generalization through BatchNorm, global average pooling, and residual learning

**Comparative Results:**

| Model | MSE | MAE (years) | Improvement |
|-------|-----|-------------|-------------|
| Custom CNN | 65.16 | 5.91 | Baseline |
| ResNet-18 | 49.42 | 4.88 | 24% better MSE |

---

## Key Takeaways

### Classical ML
- Regularization critical for generalization
- Approximate methods trade accuracy for speed effectively
- Feature engineering and scaling impact model performance
- Tree depth and splitting criteria require careful tuning

### Deep Learning
- Architecture and hyperparameter choices significantly impact performance
- Activation functions and optimizers must be matched to task
- VAEs require balanced loss functions for meaningful latent spaces
- Lower-dimensional representations can maintain or improve performance
- Reconstruction error effective for anomaly detection
- Transfer learning with pretrained models dramatically improves results
- Residual connections enable training of deeper networks
- Data augmentation helps model generalization

---

## Technologies Used

- **Languages:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Deep Learning:** PyTorch, Torchvision, Custom implementations of MLP, VAE, Autoencoder
- **Pretrained Models:** ResNet-18 (ImageNet)
- **Tools:** Jupyter Notebook, Kaggle (GPU acceleration)
