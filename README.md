
# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify dry bean varieties
using machine learning classification algorithms based on
physical measurements of dry bean samples.

This is a multiclass classification problem involving seven
different dry bean varieties.

## 2. Dataset Description

Dataset: UCI Dry Bean Dataset

Source: https://archive.ics.uci.edu/dataset/602/dry

Total Instances: 13,611

Input Features: 16

Target Variable: Class

Number of Classes: 7

The dataset contains numerical measurements of dry bean samples.
The target variable represents the corresponding bean variety.

## 3. Data Preparation

- Checked dataset dimensions and missing values.
- Separated input features and target variable.
- Used an 80:20 training and testing split.
- Applied stratified sampling to preserve class distribution.
- Used StandardScaler within pipelines for Logistic Regression,
  KNN and SVM.

## 4. Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Evaluation Metrics

The following metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Precision, Recall and F1 Score use macro averaging.
AUC is calculated using the One-vs-Rest approach with macro averaging.

## 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| SVM | 0.9221 | 0.9951 | 0.9353 | 0.9320 | 0.9335 | 0.9059 |

## 7. Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.14%.
It obtained a high AUC score of 0.9948.
Its Precision and Recall were slightly higher than SVM
in this experiment.

### Decision Tree Classifier

The Decision Tree achieved an accuracy of 89.20%.
Its performance was lower than Logistic Regression and KNN
in this experiment.

### K-Nearest Neighbors

KNN achieved an accuracy of 91.66%.
Its performance was close to Logistic Regression and SVM.
Feature scaling was applied before model training.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 76.39%.
Although its AUC score was 0.9672, its accuracy and other
classification metrics were lower than those of the other models.

### Support Vector Machine

SVM achieved the highest accuracy of 92.21% and the highest
AUC score of 0.9951.
It also achieved the highest MCC of 0.9059.
Its F1 Score was equal to Logistic Regression at four decimal places.

## 8. Overall Model Comparison and Conclusion

Based on the evaluation results, Support Vector Machine
performed best overall among the five implemented models.

Logistic Regression also showed strong performance,
with slightly higher Precision and Recall than SVM.
KNN achieved competitive results.

Gaussian Naive Bayes had comparatively lower accuracy,
while the Decision Tree achieved moderate performance.

The experiment demonstrates that different classification
algorithms can produce different results on the same dataset.

## 9. Streamlit Application

The Streamlit application provides:

- CSV test dataset upload.
- Selection of one of the five trained models.
- Display of Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Confusion matrix.
- Classification report.

The application loads the saved trained models and evaluates
the uploaded test dataset without retraining the models.

## 10. Project Files

- app.py
- requirements.txt
- README.md
- test_data.csv
- model_comparison_results.csv
- model/
  - logistic_regression.joblib
  - decision_tree.joblib
  - knn.joblib
  - gaussian_nb.joblib
  - svm.joblib

## 11. GitHub Repository

https://github.com/anix4632/dry-bean-ml-classification

## 12. Dataset Reference

UCI Machine Learning Repository:
Dry Bean Dataset

DOI: https://doi.org/10.24432/C50S4B
