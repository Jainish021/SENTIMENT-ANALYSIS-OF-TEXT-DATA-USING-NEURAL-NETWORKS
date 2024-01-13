# Sentiment Analysis of Text Data using Neural Networks

## Overview

This project focuses on implementing and analyzing Feedforward Neural Networks (FFNN) and Recurrent Neural Networks (RNN) for multi-class sentiment analysis on the Yelp reviews dataset. Key aspects include:

- Preprocessing text data for neural network models
- Developing FFNN and RNN architectures
- Training models using stochastic gradient descent and Adam optimizers
- Evaluating model performance by comparing training and validation accuracy
- Analyzing the impact of hyperparameters like epochs and hidden layers
- Assessing generalizability on a separate test set

## Models

- **FFNN**: 2-layer feedforward network with ReLU activations and softmax output
- **RNN**: Single hidden layer RNN with tanh activations and softmax output

## Results

- FFNN achieved the best validation accuracy of 56%
- RNN achieved a validation accuracy of 48%
- Test accuracy was 54% for FFNN and 44% for RNN
- Increasing epochs led to overfitting in FFNN
- RNN was more robust to overfitting due to early stopping

## Installation

The code was written in Python, and key libraries used include:

- PyTorch
- Numpy
- Matplotlib
- Sklearn
