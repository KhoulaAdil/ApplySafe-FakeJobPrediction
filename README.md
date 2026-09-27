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

## Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical variables
- Engineering structural features from job postings
- Combining and cleaning textual fields for the neural-network pipeline
- Stratified train-test splitting to preserve class distribution

For the classical machine learning experiments, **13 structural and engineered features** are used:

- telecommuting
- has_company_logo
- has_questions
- employment_type_enc
- required_experience_enc
- required_education_enc
- has_salary
- has_company_profile
- has_requirements
- has_benefits
- has_department
- text_length
- title_length

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

This comparison evaluates how different approaches to class imbalance affect the detection of fraudulent job postings.

### 2. Stratified 10-Fold Cross-Validation

The best-performing classical model, **Random Forest**, is further evaluated using **Stratified 10-Fold Cross-Validation** on the original imbalanced dataset.

Stratification preserves the proportion of real and fraudulent postings across each fold.

**Average cross-validation accuracy: 96.79%**

### 3. GloVe + Sequential Neural Network

A text-based neural network is trained using job-posting text.

**Pipeline:**

Text → Cleaning → Tokenization → Padding → GloVe Embeddings → Global Average Pooling → Dense Layer → Fraud Probability

Configuration:

* GloVe: **100-dimensional pretrained embeddings**
* Vocabulary size: **20,000**
* Maximum sequence length: **200**
* Frozen pretrained embedding layer
* Global Average Pooling
* Dense layer with ReLU activation
* Dropout regularization
* Sigmoid output layer
* SMOTE applied to the training set
* Binary cross-entropy loss
* Early stopping
* TensorFlow/Keras

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

Random Forest achieved the strongest fraudulent-class F1-score among the evaluated classical models on the original class distribution.

### Random Forest — Class Imbalance Comparison

| Training Strategy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|
| Original Distribution | 76.72% | 51.45% | 61.59% |
| Class Weighted | 76.92% | 52.02% | **62.07%** |
| SMOTE | 50.93% | **63.58%** | 56.56% |

Class weighting produced the highest F1-score for Random Forest, while SMOTE increased fraudulent-class recall to **63.58%** at the cost of lower precision.

### Random Forest — Stratified 10-Fold Cross-Validation

| Metric | Result |
|---|---:|
| Average Accuracy | **96.79%** |

The cross-validation experiment evaluates the Random Forest model across 10 stratified folds using the original imbalanced class distribution.

### GloVe Sequential Neural Network

| Metric               | Result |
| -------------------- | -----: |
| Test Loss            | 0.1727 |
| Test Accuracy        | 93.60% |
| Test AUC             | 90.20% |
| Fraudulent Precision |    40% |
| Fraudulent Recall    |    68% |
| Fraudulent F1        |    51% |

The GloVe-based neural network achieved **68% recall** for fraudulent postings, detecting a larger proportion of fraudulent samples while producing lower precision than the best-performing Random Forest configuration.

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

In particular:

- **Precision** measures how many postings predicted as fraudulent are actually fraudulent.
- **Recall** measures how many actual fraudulent postings are successfully detected.
- **F1-score** balances precision and recall.

## Key Findings

- Random Forest achieved the strongest fraudulent-class F1-score among the evaluated classical models.
- Class weighting slightly improved Random Forest's fraud F1-score from **61.59% to 62.07%**.
- SMOTE increased Random Forest fraud recall from **51.45% to 63.58%**, but reduced precision and overall F1-score.
- Stratified 10-fold cross-validation produced an average Random Forest accuracy of **96.79%** on the original class distribution.
- The GloVe-based neural network achieved **93.60% test accuracy**, **90.20% AUC**, and **68% fraudulent-class recall**.
- The experiments demonstrate why fraud-focused metrics are important when evaluating highly imbalanced classification problems.

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
