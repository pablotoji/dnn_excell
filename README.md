# Excel-Based Handwritten Digit Recognition DNN

## Overview

This project demonstrates how to integrate a Python-trained deep neural network (DNN) with an Excel-based implementation for recognizing handwritten digits. Users can "draw" a digit on an Excel 10×10 grid by marking cells with an "x". The project then replicates the forward pass of a neural network—trained on a downscaled MNIST dataset—using Excel formulas to predict the drawn digit (0-9).

## Video in Youtube

https://youtu.be/Iz2YN-1ezzM


## Features

- **Data Preprocessing:**  
  Downscale MNIST 28×28 images to 10×10 binary matrices.
  
- **DNN Training:**  
  Train a simple DNN with one hidden layer (ReLU activation) and an output layer (softmax) using TensorFlow/Keras.
  
- **Excel Integration:**  
  Generate an Excel workbook that includes:
  - **Sheet1:**  
    A 10×10 grid for user input, a flattened input helper column, and a cell that displays the predicted digit.
  - **Weights & Biases Sheets:**  
    Sheets that hold the trained weights and biases for the hidden and output layers.
  - **HiddenLayer & OutputLayer Sheets:**  
    Sheets that implement the forward pass using Excel formulas (including activation functions).

## Requirements

- Python 3.x  
- TensorFlow (version 2.x recommended)  
- Openpyxl  
- NumPy


