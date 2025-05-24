# 💳 Credit Card Fraud Detection Project

A machine learning project to detect fraudulent transactions using the **Credit Card Fraud Detection Dataset** from Kaggle. This project was developed during a 2-week industrial attachment at **Brain Station 23** and showcases the full ML pipeline from data exploration to model deployment using a **Streamlit web app**.

---

## 📌 Project Summary

- **Objective**: Accurately detect fraudulent credit card transactions.
- **Tools**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit
- **Models Used**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes
- **Challenges**: Handling severe class imbalance (~0.17% fraud cases)
- **Outcome**: Achieved ROC-AUC ~0.98 with Gradient Boosting; deployed a Streamlit-based fraud prediction app.

---

## 📅 Project Timeline

| Day   | Task Description                          |
|--------|-------------------------------------------|
| Day 2 | Data Exploration & Visualization           |
| Day 3 | Model Training & Evaluation                |
| Day 4 | Feature Engineering & Model Tuning         |
| Day 5 | Streamlit App Development & Testing        |


---

## 📁 Dataset Information

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 anonymized features (`V1`–`V28`, `Time`, `Amount`)
- **Target**: `Class` — 1 for Fraud, 0 for Normal

---

## 📊 Day 2 – Data Exploration & Visualization

### ✅ Tasks Completed:
- Loaded dataset
- Checked for null values and class imbalance
- Visualized:
  - Class distribution
  - Feature histograms and boxplots
  - Correlation heatmap
  - Log-scaled fraud vs. amount plot

### 🔍 Key Findings:
- Fraudulent transactions are extremely rare (~0.17%)
- Features `V14`, `V17`, and `V12` show strong correlation with fraud
- Fraud amounts are typically lower than normal transactions

---

## 🤖 Day 3 – Model Training & Evaluation

### ✅ Models Trained:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes

### 📈 Evaluation Metrics:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Curve

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.99     | 0.85      | 0.63   | 0.72     | 0.94    |
| Decision Tree       | 0.99     | 0.78      | 0.72   | 0.75     | 0.93    |
| KNN                 | 0.99     | 0.82      | 0.66   | 0.73     | 0.92    |
| Naive Bayes         | 0.97     | 0.24      | 0.82   | 0.37     | 0.84    |

---

## 🛠️ Day 4 – Feature Engineering & Model Tuning

### ✅ Improvements:
- Scaled `Time` and `Amount` using `StandardScaler`
- Handled class imbalance using `class_weight='balanced'`
- Hyperparameter tuning with `GridSearchCV` for:
  - Random Forest
  - Gradient Boosting
  - Baseline Decision Tree

### 🏆 Best Random Forest Parameters:
```json
{'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
````
### 🌐 Day 5 – Streamlit App Development & Testing

### ✅ Tasks Completed:
- Saved trained model using `joblib`
- Developed a **Streamlit app (`app.py`)** to:
  - Upload and process CSV files
  - Predict fraud status for each transaction
  - Display predictions with conditional styling
  - Export results to a downloadable CSV file
- Ensured consistent preprocessing (scaling, class handling) with training pipeline


### ▶️ How to Run the App

1. **Install required packages**:
```bash
pip install -r requirements.txt



