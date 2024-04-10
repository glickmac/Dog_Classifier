# Dog Breed Classifier using Convolutional Neural Networks (CNNs)

![Dog Classifier](https://github.com/glickmac/Dog_Classifier/data/dog_classifier.jpg)

## Overview

This repository contains code for building a deep-learning model that classifies dog breeds from images. The model can be accessed and a sample can be added using [Streamlit](https://dogclassifier.streamlit.app/). The model is based on Convolutional Neural Networks (CNNs) implemented using TensorFlow and Keras.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

## Introduction

Convolutional Neural Networks (CNNs) are powerful tools for image classification. In this project, we'll develop an algorithm to detect dog breeds from input images. There is no non-dog class so if a human face is detected, the algorithm will estimate the most resembling dog breed.

## Requirements

Make sure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV (for face detection)

## Dataset

We used the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) for training and evaluation. The dataset contains images of 120 different dog breeds.

## Model Architecture

Our CNN model consists of the following layers:

1. **Input Layer**: Accepts image data.
2. **Convolutional Layers**: Detects features in the image.
3. **Pooling Layers**: Reduces spatial dimensions.
4. **Fully Connected Layers**: Makes predictions.

![CNN Architecture](https://github.com/glickmac/Dog_Classifier/data/cnn_architecture.jpg)

## Training

1. Preprocess the dataset (resize, normalize, augment).
2. Train the model using labeled dog breed images.
3. Fine-tune the model using transfer learning (e.g., pre-trained ResNet-50).

## Evaluation

Evaluate the model on a validation set. Calculate accuracy and other relevant metrics.

## Usage

1. Clone this repository: `git clone https://github.com/glickmac/Dog_Classifier.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the classifier: `python dog_classifier.py <image_path>`

## Acknowledgments

- [Towards Data Science](https://towardsdatascience.com/build-your-first-deep-learning-classifier-using-tensorflow-dog-breed-example-964ed0689430) for inspiration and guidance.
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) for providing the dog breed images.

---

Feel free to add more details, instructions, and credits as needed. Good luck with your dog breed classifier! 🐶📸

(1) Build Your First Deep Learning Classifier using TensorFlow: Dog Breed .... https://towardsdatascience.com/build-your-first-deep-learning-classifier-using-tensorflow-dog-breed-example-964ed0689430.
(2) An Image Classifier on Dog Breeds using Deep Learning. https://github.com/shiinamars/image-classifier-on-dog-breeds/blob/main/README.md.
(3) Image Classifier 101: A Dog Breed Example - Medium. https://medium.com/@yolanda091107/image-classifier-101-a-dog-breed-example-cd96a1038a52.
