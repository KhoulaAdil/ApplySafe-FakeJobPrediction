# ApplySafe-FakeJobPrediction

**ApplySafe** detects **fraudulent job postings** using both **Machine Learning** and **Deep Learning** techniques.  
The project leverages the **Employment Scam Aegean Dataset (EMSCAD)** — containing ~18,000 job listings labeled as *real* or *fake* — and builds on NLP and neural network-based research methodologies for scam detection.

## Approaches Implemented

### 1. **Sequential Neural Network (SNN) with GloVe Embeddings**
- Word embeddings: **GloVe (Global Vectors for Word Representation)**  
- Architecture: Embedding → LSTM → Dense Layers  
- Optimizer: Adam | Loss: Binary Cross-Entropy  
- **Handles semantic relationships in job descriptions**

**Results**
- Test Loss: **0.05**  
- Accuracy: **98.43%**  
- AUC Score: **99.72%**

---

### 2. **Traditional Machine Learning Models**

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|----------|-----------|
| K-Nearest Neighbors (KNN) | 95.25% | 94.50% | 95.25% | 93.38% |
| Random Forest | 95.05% | 93.42% | 95.05% | 93.33% |
| Support Vector Machine (SVM) | 94.94% | 90.13% | 94.94% | 92.47% |
| Decision Tree | 94.94% | 93.10% | 94.94% | 93.30% |
| Naive Bayes | 92.31% | 90.87% | 92.31% | 91.56% |
| Multilayer Perceptron (MLP) | 95.11% | 95.35% | 95.11% | 92.88% |

**Highlights**
- Excellent overall performance (Accuracy > 94%)  
- **Random Forest** and **KNN** were top traditional models  

---

### 3. **Deep Neural Network (DNN) with SMOTE & Stratified K-Fold**
- **SMOTE** used for synthetic oversampling of minority (fraudulent) class  
- **Stratified K-Fold Cross-Validation** ensures balanced splits  
- Custom architecture tuned for text classification

**Results (Average across folds)**
- Accuracy: **77.38%**  
- Precision: **15.61%**  
- Recall: **83.25%**  

> Despite lower accuracy, the **DNN achieved the highest recall**, making it the most effective at identifying *fraudulent* posts — a critical goal in job scam detection.

---

## Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC  
- Confusion Matrix  
- Cross-validation averages  

---

## Technologies Used
- **Python**, **TensorFlow/Keras**, **Scikit-learn**, **NLTK**  
- **GloVe embeddings** for semantic representation  
- **SMOTE**, **Stratified K-Fold** for data balancing and reliability  
- **Matplotlib**, **Seaborn** for visualization 
