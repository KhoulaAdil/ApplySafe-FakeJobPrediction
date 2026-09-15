# ApplySafe — Fake Job Prediction

**ApplySafe** is a machine learning and deep learning project for detecting fraudulent job postings using the **Employment Scam Aegean Dataset (EMSCAD)**.

The project evaluates classical machine learning models and neural-network-based approaches for binary classification of job postings as **Real** or **Fraudulent**.

## Dataset

**EMSCAD (Employment Scam Aegean Dataset)**

* **17,880** job postings
* **17,014** real postings
* **866** fraudulent postings
* **18** attributes
* Fraudulent class: **4.84%**

The dataset contains job-related textual, categorical, and binary attributes, including job title, company profile, description, requirements, benefits, employment type, required experience, education, industry, and function.

## Models

### 1. Classical Machine Learning

The following models are evaluated:

* K-Nearest Neighbors (KNN)
* Random Forest
* Decision Tree
* Support Vector Machine (SVM)
* Naive Bayes
* Multilayer Perceptron (MLP)

The experiments compare:

* Original class distribution
* Class-weighted training
* SMOTE oversampling

### 2. GloVe + Sequential Neural Network

A text-based neural network is trained using job-posting text.

**Pipeline:**

Text → Tokenization → Padding → GloVe Embeddings → Neural Network → Fraud Probability

Configuration:

* GloVe: **100-dimensional pretrained embeddings**
* Vocabulary size: **20,000**
* Maximum sequence length: **200**
* Binary classification
* TensorFlow/Keras

### 3. DNN with SMOTE and Stratified K-Fold

A neural-network approach is evaluated using:

* **SMOTE** for minority-class oversampling
* **Stratified K-Fold Cross-Validation**
* Fraud-focused evaluation using precision, recall, and F1-score

## Results

### Classical ML — Original Class Distribution

| Model             |   Accuracy |  Precision | Recall |         F1 |
| ----------------- | ---------: | ---------: | -----: | ---------: |
| KNN               |     96.09% |     76.19% | 27.75% |     40.68% |
| **Random Forest** | **96.90%** | **76.72%** | 51.45% | **61.59%** |
| Decision Tree     |     95.67% |     55.17% | 55.49% |     55.33% |
| SVM               |     95.16% |      0.00% |  0.00% |      0.00% |
| Naive Bayes       |     87.47% |     21.17% | 58.38% |     31.08% |
| MLP               |     94.63% |     42.28% | 30.06% |     35.14% |

### GloVe Sequential Neural Network

| Metric               | Result |
| -------------------- | -----: |
| Test Loss            | 0.1727 |
| Test Accuracy        | 93.60% |
| Test AUC             | 90.20% |
| Fraudulent Precision |    40% |
| Fraudulent Recall    |    68% |
| Fraudulent F1        |    51% |

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Stratified Cross-Validation

Because fraudulent postings represent only **4.84%** of the dataset, fraudulent-class **recall and F1-score** are considered alongside overall accuracy.

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **imbalanced-learn**
* **NLTK**
* **GloVe**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **WordCloud**
* **Joblib**
