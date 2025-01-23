# ML-assigment
This repository contains  asisgnment as a part of Machine Learning , on various Linear and Ragression models used for data classification. 
The task ois based upon this problem :https://docs.google.com/document/d/1CDXyJNWyVxSf37nwdh3eIgHj9-5Wqw_-2-nRaYS33g0/edit?usp=drivesdk
This assignment involves loading, preprocessing, and training multiple machine learning models to classify breast cancer data using the scikit-learn library. Here's a summary of the key steps:

Loading and Preprocessing Data:
Inbuilt Breast Cancer dataset in Python is made use of here.

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
Overall, the assignment demonstrates how to preprocess data, train multiple machine learning classifiers, and evaluate their performance on a classification task.  The code built is attached within this repository.



KEY NOTES
1. Logistic Regression:
Logistic regression is a linear classifier used primarily for binary classification problems. It works by estimating the probability that an instance belongs to a particular class using the logistic (sigmoid) function. The function maps input values to a range between 0 and 1, which is interpreted as the probability of a class. The model outputs a value close to 0 for one class and close to 1 for the other class. Logistic regression is simple, efficient, and performs well for problems with linear decision boundaries. However, it struggles when the data exhibits complex, non-linear relationships. It also requires careful handling when the features are correlated or when the data is imbalanced.

2. Random Forest Classifier:
Random Forest is an ensemble learning method that builds multiple decision trees during training and combines their results for classification. Each tree is trained on a random subset of the data, using a random subset of features to split at each node. This reduces overfitting and increases model robustness. Random Forest classifiers can capture both linear and non-linear relationships between features, making them highly versatile. They also handle missing data and outliers better than individual decision trees. The main downside is that they can be computationally expensive and require tuning for optimal performance, especially when working with a large number of trees or high-dimensional data.

3. Support Vector Machine (SVM):
The Support Vector Machine is a powerful classification algorithm that works by finding the hyperplane that best separates data points of different classes in a higher-dimensional space. For non-linearly separable data, SVM uses kernel tricks (such as the radial basis function) to project the data into a higher dimension where a linear separation is possible. SVMs are particularly effective in high-dimensional spaces and situations where there is a clear margin of separation. However, they can be sensitive to the choice of kernel, regularization parameters, and the size of the dataset. SVMs tend to be computationally expensive, especially for large datasets, and may not scale well without careful parameter tuning.

4. K-Nearest Neighbors (KNN):
K-Nearest Neighbors is an instance-based learning algorithm where classification is done based on the majority class of a sample’s 'k' closest training instances in the feature space. KNN does not require a training phase, making it simple to implement. It is highly intuitive, as it classifies a new data point based on the similarity (distance) to other data points. However, KNN can become computationally expensive during prediction, as it needs to compute distances to every training sample. The algorithm is sensitive to the choice of distance metric and the number of neighbors, and it can be inefficient when dealing with high-dimensional data due to the curse of dimensionality, which makes distance calculations less meaningful.

Comparison and Summary:
Each of the classifiers has its own set of strengths and weaknesses. Logistic Regression is fast and works well for linear problems but can struggle with more complex data. Random Forests offer high flexibility and perform well with non-linear data, but they require more computational resources. SVMs are effective for complex data and high-dimensional spaces but need careful tuning and are computationally intensive. KNN is simple and intuitive but can be slow for large datasets and high dimensions. The best model for a given task depends on the nature of the data, the problem complexity, and computational constraints.








