# ML-assigment
This repository contains  asisgnment as a part of Machine Learning , on various Linear and Ragression models used for data classification. 

This assignment involves loading, preprocessing, and training multiple machine learning models to classify breast cancer data using the scikit-learn library. Here's a summary of the key steps:

Loading and Preprocessing Data:

The Breast Cancer dataset from sklearn.datasets is loaded. This dataset contains features about cell characteristics of malignant and benign tumors.
The features (X) are scaled using StandardScaler to standardize the data to have a mean of 0 and a standard deviation of 1.
The dataset is then split into training and test sets using train_test_split, with 80% for training and 20% for testing.
Model Training and Evaluation: Several classifiers are trained and evaluated using the scaled data:

Logistic Regression:
A logistic regression model is trained on the training set.
Accuracy is computed on the test set.
Predictions are made, and results are compared with true labels.


Random Forest Classifier:
A random forest classifier is trained with 100 estimators and a maximum depth of 5.
Accuracy is evaluated, and sample predictions are displayed.

Support Vector Machine (SVM):
A linear SVM classifier is trained on the scaled data.
The accuracy and predictions are evaluated and displayed.

K-Nearest Neighbors (KNN):
A KNN classifier with 5 neighbors is trained.
The accuracy and predictions are evaluated and displayed.
Performance Evaluation:

For each model, the accuracy score is computed on the test set.
The first 10 predictions are displayed for each model, showing the true label and predicted class.
Overall, the assignment demonstrates how to preprocess data, train multiple machine learning classifiers, and evaluate their performance on a classification task.
