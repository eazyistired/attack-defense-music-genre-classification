# Adversarial Music Genre Classification

This project implements a baseline **Convolutional Neural Network (CNN)** for music genre classification and evaluates its robustness against a **Projected Gradient Descent (PGD)** adversarial attack.

***

## Setup and Running the Code

The code is provided as a **Jupyter Notebook (`AI3_Project.ipynb`)** and is designed to run in a **Google Colab** environment, utilizing a **GPU (CUDA)** for training.

### Prerequisites
* A Google account to use **Google Colab**.
* The **GTZAN Dataset** is required. The notebook uses the `kagglehub` library to automatically download the dataset from Kaggle. Ensure your environment has the necessary permissions/setup for Kaggle access if running outside a standard Kaggle/Colab setup.

### Project Steps
1.  **Mount Google Drive** and set up the project folder structure.
2.  **Download and Prepare Data**: The notebook automatically downloads the **GTZAN dataset** and splits it into training, validation, and test sets (70%/15%/15% split).
3.  **Model Training**: The **`AudioCNN`** model is defined and trained using the prepared dataset.
4.  **Evaluation**: The model is evaluated for its **clean accuracy** and then its **adversarial accuracy** using the PGD attack.

***

## Algorithms and Methods Used

### 1. Classification Model
* **Architecture**: A simple **Convolutional Neural Network (CNN)** (`AudioCNN`) is used, consisting of multiple convolutional, pooling, and batch normalization layers, followed by a fully connected classifier.
* **Feature Extraction**: The model operates on **raw audio signals**, converting them internally into a **Mel Spectrogram** representation.

### 2. Adversarial Attack
* **Method**: The **Projected Gradient Descent (PGD) attack with L2 norm constraint** (`pgd_attack_l2`) is used to generate powerful adversarial examples by iteratively moving in the direction of the loss gradient.
* **Parameters**: The attack utilized an $\mathbf{\epsilon}$ (maximum perturbation distance) of **2.0** and **20 steps** with an $\mathbf{\alpha}$ (step size) of **0.2**.

***

## Additional Notes

* **Dataset**: The project uses the **GTZAN dataset** to classify audio clips into **10 music genres**.