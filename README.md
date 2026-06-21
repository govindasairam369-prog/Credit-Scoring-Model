# 💳 Credit Scoring Model using Machine Learning

## 📌 Project Overview

This project predicts whether a loan applicant is likely to default on a loan using machine learning classification algorithms. By analyzing financial and personal information, the model helps assess an applicant's creditworthiness.

## 🎯 Objective

The objective of this project is to build and compare multiple machine learning models for credit risk prediction and identify the best-performing model using evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 📂 Dataset

The project uses a credit risk dataset containing information about loan applicants, including:

* Person Age
* Person Income
* Home Ownership
* Employment Length
* Loan Intent
* Loan Grade
* Loan Amount
* Loan Interest Rate
* Loan Percent Income
* Previous Loan Default History
* Credit History Length

**Target Variable:**

* `loan_status`

  * `0` → Low credit risk
  * `1` → High credit risk / Loan default

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib
* VS Code
* Git & GitHub

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 🏆 Results

| Model               | Accuracy   | Precision  | Recall     | F1 Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 86.76%     | 76.80%     | 56.33%     | 64.99%     | 86.94%     |
| Decision Tree       | 88.66%     | 72.63%     | 77.07%     | 74.79%     | 84.48%     |
| **Random Forest**   | **93.37%** | **97.05%** | **71.80%** | **82.54%** | **92.92%** |

### ✅ Best Performing Model

**Random Forest** achieved the best overall performance with:

* Accuracy: **93.37%**
* Precision: **97.05%**
* F1 Score: **82.54%**
* ROC-AUC: **92.92%**

---

## ⚙️ Project Workflow

1. Load and inspect the dataset
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features
5. Split data into training and testing sets
6. Train multiple classification models
7. Evaluate model performance
8. Compare results and select the best model

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Model deployment using Flask or FastAPI
* Interactive web interface for predictions

---

## 📁 Project Structure

```text
Credit_Scoring_Model/
│
├── data/
│   └── credit_risk_dataset.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 👨‍💻 Author

**Govinda Sai Ram**

This project was developed as part of my machine learning learning journey to gain hands-on experience in data preprocessing, model training, evaluation, and credit risk prediction.
