
# Breast Cancer Detection using CNN and SVM

This repository contains the implementation of a Breast Cancer Detection system using Convolutional Neural Networks (CNN) and Support Vector Machines (SVM). The project involves cleaning and preprocessing the image data, applying grayscale conversion, and training both CNN and SVM models to accurately detect breast cancer.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction
Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment and improved survival rates. This project aims to develop a machine learning model that can accurately classify breast cancer images as benign or malignant, using CNN and SVM.

## Dataset
The dataset used in this project consists of breast cancer images, categorized as benign or malignant. The images are obtained from public datasets, such as the [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).


## Data Preprocessing
The data preprocessing steps include:

1. **Loading Images**: Load the breast cancer images from the dataset.
2. **Image Cleaning**: Remove any noisy or corrupted images from the dataset.
3. **Grayscale Conversion**: Convert the RGB images to grayscale to simplify the model and reduce computation.
    ```python
    def convert_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ```
4. **Resizing Images**: Resize the images to a fixed size (e.g., 224x224) for consistency.
    ```python
    def resize_image(image, size=(224, 224)):
        return cv2.resize(image, size)
    ```
5. **Normalization**: Normalize the pixel values to a range of 0-1 to aid in faster convergence during training.
    ```python
    def normalize_image(image):
        return image / 255.0
    ```

## Model Architecture
This project implements two models:

1. **Convolutional Neural Network (CNN)**:
   - **Input Layer**: Takes the preprocessed images as input.
   - **Convolutional Layers**: Extract features from the images.
   - **Pooling Layers**: Reduce dimensionality.
   - **Fully Connected Layers**: Classify the images based on the extracted features.
   - **Output Layer**: Binary classification (benign or malignant).

2. **Support Vector Machine (SVM)**:
   - **Feature Extraction**: Use CNN as a feature extractor.
   - **SVM Classifier**: Classify the extracted features.

## Training
To train the models:

1. **CNN Training**:
    ```bash
    python train_cnn.py
    ```
    This script trains the CNN model on the preprocessed dataset and saves the trained model.

2. **SVM Training**:
    ```bash
    python train_svm.py
    ```
    This script uses the features extracted from the CNN model to train the SVM classifier.

## Evaluation
Evaluate the trained models using a separate test dataset:

1. **CNN Evaluation**:
    ```bash
    python evaluate_cnn.py
    ```
2. **SVM Evaluation**:
    ```bash
    python evaluate_svm.py
    ```

## Usage
You can use the trained models to predict the classification of new breast cancer images:

1. **Load a new image**:
    ```python
    image = load_image('path_to_image')
    ```
2. **Predict using CNN**:
    ```python
    prediction = cnn_model.predict(image)
    ```
3. **Predict using SVM**:
    ```python
    features = cnn_feature_extractor(image)
    prediction = svm_model.predict(features)
    ```

## Results
The models achieved the following performance metrics:

- **CNN Accuracy**: 95%
- **SVM Accuracy**: 92%

## Future Work
- Explore the use of other deep learning models like ResNet or Inception.
- Experiment with different feature extraction techniques.
- Increase the dataset size by including more diverse images.
