# 📊 Bank Marketing Campaign – Binary Classification

## 🎯 Project Objective

The objective of this project is to **predict whether a client will subscribe to a term deposit** based on demographic information and marketing campaign data.

This is framed as a **binary classification problem**, where:
- `1` indicates that the client **subscribed** to the service
- `0` indicates that the client **did not subscribe**

Given the business context, **recall for the positive class (subscribers)** is considered a key evaluation metric, as identifying potential clients is more valuable than minimizing false positives.

---

## 📁 Dataset

The dataset was obtained from **Kaggle** and contains information related to direct marketing campaigns conducted by a banking institution.

https://www.kaggle.com/c/bank-marketing-uci

It includes:
- Client demographic features (age, job, education, etc.)
- Financial information (housing loan, personal loan, balance)
- Campaign-related variables (number of contacts, previous outcomes)
- Target variable `y`

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA stage focused on:
- Understanding the structure of the dataset
- Identifying numerical and categorical variables
- Analyzing the distribution of the target variable `y`
- Detecting class imbalance
- Exploring dominant categories in categorical features
- Checking for missing values and data quality issues

A strong **class imbalance** was observed, with a significantly smaller proportion of clients subscribing to the service. This motivated:
- The use of **precision and recall** as primary metrics
- Class weighting during model training
- Stratified cross-validation

---

## ⚙️ Preprocessing

Data preprocessing was performed using a `ColumnTransformer`, including:
- **Standardization** of numerical features
- **One-hot encoding** of categorical variables
- Fitting the preprocessing pipeline **only on training data** to avoid data leakage

The same preprocessing pipeline was consistently applied during cross-validation.

---

## 🤖 Models Implemented

The following models were trained and evaluated:

1. **Decision Tree**
   - With and without pruning
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest**
4. **Deep Neural Network (DNN)** using TensorFlow / Keras

For the most promising models (Random Forest and DNN), **hyperparameters were tuned manually** and class imbalance was handled using `class_weight`.

---

## 🔁 Cross-Validation Strategy

To ensure fair and comparable evaluation:
- **Stratified K-Fold Cross-Validation** was used
- The **same folds** were applied to both Random Forest and DNN
- Metrics were computed **per fold** and aggregated

The evaluation focused on:
- **Precision (class 1)**
- **Recall (class 1)**

---

## 📈 Cross-Validation Results

| Model         | Precision (mean) | Precision (std) | Recall (mean) | Recall (std) |
|---------------|------------------|------------------|---------------|---------------|
| Random Forest | 0.389            | 0.020            | 0.772         | 0.048         |
| DNN           | 0.318            | 0.052            | 0.827         | 0.077         |

---

## 🧠 Final Conclusions

- The **Random Forest** model achieves higher **precision**, meaning its positive predictions are more reliable.
- The **DNN** model achieves higher **recall**, identifying a larger proportion of potential subscribers.
- Given the business goal of **identifying as many potential clients as possible**, recall was prioritized over precision.
- Therefore, despite lower precision, the **DNN model is considered more suitable** for this problem.

Additionally, threshold tuning was applied to improve precision while maintaining acceptable recall levels.

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Jupyter Notebook

---

## ✍️ Author

**Sebastian Souto**

