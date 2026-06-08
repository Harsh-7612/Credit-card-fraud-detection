# 🕵️‍♂️ Credit Card Fraud Detection Project

This repository contains a comprehensive machine learning pipeline designed to identify fraudulent credit card transactions. Due to the extreme class imbalance inherent in fraud datasets, the project employs advanced resampling techniques and robust evaluation metrics beyond simple accuracy to build highly reliable detection models.

## 🚀 Key Features
* **Exploratory Data Analysis (EDA):** In-depth structural analysis and statistical profiling of transaction records.
* **Class Imbalance Mitigation:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to prevent model bias toward majority normal transactions.
* **Multi-Model Pipeline:** Built, tuned, and compared Logistic Regression, Random Forest, and XGBoost classifiers.
* **Comprehensive Metrics Evaluation:** Validation using Classification Reports, Confusion Matrices, ROC-AUC Curves, and Precision-Recall Curves.

---

## 📊 Dataset Overview
The project utilizes anonymized European cardholder transaction data containing 284,807 entries.
* **Features:** 31 numerical features, including `Time`, `Amount`, and 28 principal components (`V1` through `V28`) obtained via PCA.
* **Missing Values:** 0 null entries across all features, ensuring full structural integrity.

### ⚖️ The Class Imbalance Problem
Fraudulent transactions represent a tiny fraction of total activity:
* **Normal Class (`0`):** 284,315 transactions (~99.83%)
* **Fraudulent Class (`1`):** 492 transactions (~0.17%)

### 💡 Statistical Observations
* **Normal Transactions Mean:** \$88.29
* **Fraudulent Transactions Mean:** \$122.21
* This difference shows that fraudulent behaviors tend to shift toward higher average transaction magnitudes.

---

## 🛠️ Data Preprocessing & Balancing

1. **Train-Test Split:** A stratified 80/20 split preserves original class proportions across training and testing sets.
2. **SMOTE Resampling:** Synthetic fraud entries are generated within the training set to prevent classifiers from simply ignoring the minority class:
   * **Pre-SMOTE Training Size:** 227,845 records
   * **Post-SMOTE Training Size:** 454,902 records (perfectly balanced with 227,451 samples per class)

---

## 🤖 Models & Performance Summary

Three distinct classification algorithms were trained on the balanced data and validated against the original, unbalanced test partition:

| Model | Test Accuracy | ROC-AUC Score |
| :--- | :---: | :---: |
| **Random Forest Classifier** | **99.95%** | 0.980691 |
| **XGBoost Classifier** | 0.9994% | 0.981295 |
| **Logistic Regression** | 0.9899% | **0.989152** |

### 📈 Key Insights
* **Random Forest** achieved the highest overall test accuracy (**99.95%**), demonstrating excellent boundary classification.
* **Logistic Regression** achieved the highest overall **ROC-AUC score (0.9892)**, providing strong structural reliability across different probability thresholds.

---

## 📂 Project Structure
```text
├── ccfraud-detection.ipynb    # Main development, analysis, and visualization notebook
├── creditcard.csv             # Raw transaction dataset (to be downloaded separately)
└── README.md                  # Project documentation
