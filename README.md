# Adversarial Music Genre Classification

This project implements a baseline **Convolutional Neural Network (CNN)** for music genre classification and evaluates its robustness against adversarial attacks.

***

## Setup and Running the Code

The code is provided as a **Jupyter Notebook (`AI3_Project.ipynb`)** and is designed to run in a **Google Colab** environment, utilizing a **GPU (CUDA)** for training.

### Prerequisites
* A Google account to use **Google Colab**.
* The **GTZAN Dataset** is required. The notebook uses the `kagglehub` library to automatically download the dataset from Kaggle.

### Project Steps
1.  **Mount Google Drive** and set up the project folder structure.
2.  **Download and Prepare Data**: The notebook automatically downloads the **GTZAN dataset** and splits it into training, validation, and test sets (70%/15%/15% split).
3.  **Model Training**: The **`AudioCNN`** model is defined and trained using the prepared dataset.
4.  **Evaluation**: The model is evaluated for its **clean accuracy** and then its **adversarial accuracy** using the PGD attack.

***

## Algorithms and Methods Used

### 1. Classification Model
* **Architecture**: A simple **Convolutional Neural Network (CNN)** (`AudioCNN`) used for 10-genre classification.
* **Feature Extraction**: The model operates on **raw audio signals**, converting them internally into a **Mel Spectrogram** representation.

### 2. Adversarial Attacks
* **Projected Gradient Descent (PGD) Attack**: The primary iterative, gradient-based attack used in the notebook. It generates adversarial examples by taking small steps in the direction of the loss gradient, but stays within a maximum allowed perturbation ($\epsilon$) using an **L2 norm constraint**.
* **Carlini & Wagner (C&W) Attack**: A stronger, optimization-based attack often used as a benchmark. It is designed to find the **minimum possible perturbation** ($\delta$) required to cause a misclassification by solving a formal optimization problem. This makes it highly effective against many defenses.

### 3. Defense Mechanisms
* **Adversarial Training**: A powerful defense mechanism where the model is periodically or continuously retrained using a mixture of **clean data and generated adversarial examples**. This explicitly teaches the model to classify perturbed inputs correctly, increasing its resilience.
* **Lipschitz Regularization**: A defense strategy that improves robustness by **constraining the rate of change** of the model's output with respect to its input. This is done during training by minimizing a regularization term that limits the model's **Lipschitz constant** (or the norm of the input gradient), effectively "smoothing" the decision boundary and reducing the model's sensitivity to small perturbations.

***