# 💳 Credit Card Fraud Detection Project

A machine learning project to detect fraudulent transactions using the **Credit Card Fraud Detection Dataset** from Kaggle. This project spans multiple phases from **exploratory data analysis (EDA)** to **model tuning**, conducted during a 2-week industrial attachment at **Brain Station 23**.

---

## 📅 Project Timeline

| Day | Task Description |
|-----|------------------|
| Day 2 | Data Exploration & Visualization |
| Day 3 | Model Training & Evaluation |
| Day 4 | Feature Engineering & Model Improvement |

---

## 📁 Dataset Info

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 anonymized features (`V1` to `V28`, `Time`, `Amount`)
- **Target**: `Class` — 1 for Fraud, 0 for Normal

---

## 📊 Day 2 – Data Exploration & Visualization

### ✅ Actions Performed:
- Loaded dataset from Google Drive
- Checked for nulls and imbalance
- Created multiple visualizations:
  - Class distribution (fraud vs. normal)
  - Histograms of key features
  - Boxplots and KDE plots
  - Correlation heatmap for top 10 features
  - Log-scaled fraud vs. amount chart

### 📌 Key Observations:
- Dataset is highly imbalanced (~0.17% fraud)
- Features `V14`, `V17`, `V12` strongly correlate with fraud
- Fraud amounts tend to be lower than normal transactions
- Time doesn’t follow a consistent fraud pattern

---

## 🤖 Day 3 – Classification Model Training & Evaluation

### ✅ Models Used:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes

### 📈 Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

### 📊 Results Snapshot:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.99 | 0.85 | 0.63 | 0.72 | 0.94 |
| Decision Tree       | 0.99 | 0.78 | 0.72 | 0.75 | 0.93 |
| KNN                 | 0.99 | 0.82 | 0.66 | 0.73 | 0.92 |
| Naive Bayes         | 0.97 | 0.24 | 0.82 | 0.37 | 0.84 |

### 📌 Insights:
- Logistic Regression performed decently with high precision
- Naive Bayes had very high recall but poor precision (too many false positives)
- ROC Curve helped visualize tradeoffs between models

---

## 🛠️ Day 4 – Feature Engineering & Model Tuning

### ✅ Enhancements:
- StandardScaler applied to `Time` and `Amount`
- Handled class imbalance using `class_weight='balanced'`
- Trained and tuned:
  - Random Forest (with `GridSearchCV`)
  - Gradient Boosting (with `GridSearchCV`)
  - Baseline Decision Tree

### 📈 Hyperparameter Tuning (GridSearchCV):

**Random Forest Best Params:**
```json
{'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
