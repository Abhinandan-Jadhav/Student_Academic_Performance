# Project Report

## Title
AI-Based Student Performance Prediction and Early Academic Risk Detection System

## Abstract
This project applies deep learning to student performance prediction. Student
academic and behavioral attributes are processed using NumPy, Pandas and
Scikit-learn. A Multilayer Perceptron is trained using TensorFlow/Keras to
classify students into At Risk, Average, Good and Excellent categories.

The project intentionally demonstrates the learning progression from basic
NumPy operations and perceptrons to activation functions, forward propagation,
backpropagation, loss functions, optimizers and a final MLP.

## Problem Statement
Manual identification of academically at-risk students from multiple academic
variables is difficult. An AI model can analyze these variables consistently
and provide an early indication of potential performance risk.

## Objectives
- Perform NumPy array operations.
- Implement dot products and a perceptron.
- Visualize activation functions and matrices.
- Implement forward and backward propagation manually.
- Demonstrate MSE and cross entropy.
- Study SGD and Adam.
- Train an MLP using Keras.
- Evaluate using Accuracy, Precision, Recall and F1-score.
- Provide a usable prediction dashboard.

## Inputs
Attendance, study hours, previous score, assignment score, internal score,
practical score, participation and GPA.

## Output
At Risk, Average, Good or Excellent, together with confidence, risk level and
recommendations.

## Methodology
Data collection -> preprocessing -> scaling -> train/test split -> MLP training
-> cross entropy loss -> Adam optimization -> evaluation -> prediction.

## Ethical Note
Predictions should not be used as the sole basis for grading, disciplinary
action or denial of educational opportunities. Validate the model on
representative real-world data and monitor for bias.
