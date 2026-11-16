# SVM vs. MLP (Hinge Loss) Classification on CIFAR-10

This repository contains the implementation, tuning, and comparative analysis of **Support Vector Machines (SVM)** and a custom **Multilayer Perceptron (MLP)** for image classification on the CIFAR-10 dataset. 

This project expands upon previous baseline models (K-Nearest Neighbors and Nearest Centroid Classifier) by introducing margin-based classification techniques, notably exploring various SVM kernels and a custom Multi-class Hinge Loss implementation in PyTorch.

## 🎯 Project Purpose

The primary objective is to evaluate how well shallow and margin-based models handle complex image data. Specifically, the project aims to:
* Train and tune SVMs with various kernels (Linear, RBF, Polynomial) and hyperparameters ($C$, $\gamma$, degrees).
* Implement an MLP from scratch using PyTorch with a custom **Crammer-Singer Multi-class Hinge Loss**.
* Apply Dimensionality Reduction (**PCA**) to optimize feature space and training time.
* Compare the final models against each other—and against previously implemented K-NN and NCC algorithms—based on accuracy, speed, and generalization ability (Overfitting/Underfitting metrics).

## 📊 Dataset & Preprocessing

The project utilizes the **CIFAR-10** dataset (50,000 training images, 10,000 test images, 10 classes). 

To adapt the 3D image data for SVMs and shallow networks, the following preprocessing pipeline is applied:
1. **Flattening:** $32 \times 32 \times 3$ images are flattened into 3072-dimensional vectors.
2. **Standardization:** `StandardScaler` is applied to ensure zero mean and unit variance ($mean \approx 0, std \approx 1$).
3. **Dimensionality Reduction (PCA):** Principal Component Analysis is applied to reduce the feature space from 3072 to **100 components**, retaining ~89.8% of the dataset's variance while significantly accelerating SVM training.

## 🧠 Models Implemented

### 1. Support Vector Machines (scikit-learn)
Extensive hyperparameter tuning was conducted across different kernel configurations:
* **Linear Kernel:** Evaluated for $C \in \{0.1, 1.0, 10.0\}$.
* **RBF Kernel:** Evaluated for $C \in \{0.1, 1.0, 10.0, 100.0\}$ and $\gamma \in \{\text{scale}, 0.01, 0.001\}$.
* **Polynomial Kernel:** Evaluated for degrees 2, 3, and 4.

### 2. MLP with Multi-Class Hinge Loss (PyTorch)
* **Architecture:** 1 Hidden Layer (128 neurons) with ReLU activation and Dropout.
* **Custom Loss Function:** Implemented `MultiClassHingeLoss` in PyTorch, which forces the model to score the correct class higher than all incorrect classes by a specified margin (analogous to the SVM objective).
* **Optimization:** Adam optimizer with optional L2 regularization (Weight Decay) and Early Stopping.

## 🛠️ Technologies & Libraries
* **PyTorch (`torch`, `torch.nn`, `torch.optim`):** For defining the MLP, custom Hinge Loss, and tensor operations.
* **scikit-learn (`sklearn.svm`, `PCA`, `StandardScaler`, `metrics`):** For SVM implementations, preprocessing, and evaluation metrics.
* **torchvision:** For downloading and handling the CIFAR-10 dataset.
* **Matplotlib:** For visualizing confusion matrices, predictions, and training curves.

## 🏆 Key Findings & Final Comparison

After evaluating all models, the results were compared alongside the previously implemented K-NN and NCC algorithms:

1. **Overall Winner - MLP (Hinge Loss):** 
   * Achieved the highest test accuracy (**~50.6%**).
   * Fast training time (~58s) utilizing mini-batch gradient descent.
   * Demonstrated the best balance of accuracy and efficiency with only slight overfitting (gap ~11.9%).
2. **Strong Runner-Up - RBF SVM ($C=1.0, \gamma=0.001$):**
   * Reached ~48.3% accuracy but suffered from severe overfitting (24% train-test gap) and was slower to train.
3. **Model Failures:**
   * Both Linear SVM (~17% accuracy) and Polynomial SVM (~13.5% accuracy) completely failed to capture the non-linear patterns of the CIFAR-10 image data, heavily underfitting the dataset.
4. **Baseline Comparisons:**
   * **NCC** remains the fastest (1.12s) but weakest (~28%).
   * **K-NN ($k=1$)** achieves decent baseline accuracy (~35.6%) but is impractically slow for inference compared to parametric models.

*Insight: CIFAR-10 is highly complex for shallow models. While PCA speeds up training immensely, it discards spatial information. The custom MLP with Hinge Loss proved to be the most robust architecture among the tested non-convolutional models.*
