# Dry Bean Classification Using Machine Learning

## 1. Problem Statement
This project uses supervised machine learning algorithms to classify dry bean varieties from physical measurements. It is a multiclass classification task involving seven varieties.

## 2. Dataset Description
- Dataset: UCI Dry Bean Dataset
- Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Records: 13,611
- Input features: 16 numerical features
- Target: `Class`
- Classes: Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira
- DOI: https://doi.org/10.24432/C50S4B
- License: CC BY 4.0

## 3. GitHub Repository
https://github.com/anix4632/dry-bean-ml-classification

## 4. Live Streamlit Application
https://dry-bean-ml-classification-yonyqxftojx99ubafqsabd.streamlit.app/

The app accepts a labelled test CSV, lets users select one of the five saved models, and displays evaluation metrics, a confusion matrix and a classification report. It uses saved models and does not retrain on uploaded test data.

## 5. Data Preparation
- Separated the 16 input features from the `Class` target.
- Used an 80:20 stratified training/testing split.
- Applied StandardScaler within pipelines for Logistic Regression, KNN and SVM.
- Evaluated all five models on the same held-out test split.

## 6. Machine Learning Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 7. Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). Precision, Recall and F1 use macro averaging. AUC uses the One-vs-Rest approach with macro averaging.

## 8. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| SVM | **0.9221** | **0.9951** | 0.9353 | 0.9320 | **0.9335** | **0.9059** |

## 9. Observations

| Model | Observation |
|---|---|
| Logistic Regression | Strong results: 0.9214 accuracy and 0.9948 AUC. It achieved the highest Precision (0.9354) and Recall (0.9321). F1 ties with SVM to four decimal places. |
| Decision Tree | Accuracy was 0.8920, lower than Logistic Regression, KNN and SVM on this test split. |
| KNN | Accuracy was 0.9166 and AUC was 0.9833, making it competitive with the leading models. Feature scaling was applied. |
| Gaussian Naive Bayes | Accuracy was 0.7639. Its AUC was 0.9672, but its other reported classification metrics were comparatively lower. |
| SVM | Highest Accuracy (0.9221), AUC (0.9951) and MCC (0.9059). F1 ties with Logistic Regression to four decimal places. |

## 10. Overall Winner and Conclusion
**SVM is the overall best-performing model in this experiment**, with the highest Accuracy, AUC and MCC. Its advantage over Logistic Regression is small (approximately two additional correctly classified test samples), so the two models performed very similarly. Logistic Regression had slightly higher Precision and Recall. KNN was also competitive; Decision Tree and Gaussian Naive Bayes had lower overall results.

## 11. Project Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
- `test_data.csv` — Held-out labelled test data
- `model_comparison_results.csv` — Model comparison metrics
- `ML_Assignment_2_Dry_Bean.ipynb` — Training and evaluation notebook
- `model/` — Saved models: `logistic_regression.joblib`, `decision_tree.joblib`, `knn.joblib`, `gaussian_nb.joblib`, `svm.joblib`

## 12. Dataset Reference
UCI Machine Learning Repository, Dry Bean Dataset. DOI: https://doi.org/10.24432/C50S4B. Licensed under CC BY 4.0.
