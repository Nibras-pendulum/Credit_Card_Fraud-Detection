# Credit Card Fraud Detection - Data Exploration & Visualization

This repository contains a Jupyter Notebook for exploring and visualizing the **Credit Card Fraud Dataset** from Kaggle. The goal is to understand the data structure, distributions, and fraud trends to support machine learning model development for fraud detection.

---

## Dataset

- Source: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Format: CSV file (`creditcard.csv`)
- Description: Contains credit card transactions labeled as fraudulent or normal.

---

## Contents

- **Data Loading & Inspection**  
  Loading the dataset using Pandas and checking for missing values and general info.

- **Visualizations**  
  - Class distribution with log scale  
  - Histograms of transaction amounts and feature distributions  
  - Correlation heatmaps for all features and top correlated features  
  - Scatter plots and boxplots to explore amount and time trends  
  - Pairplots to examine feature relationships

- **Observations & Insights**  
  Key findings related to fraud trends and data imbalances.

---

## Requirements

- Python 3.x  
- Colab Notebook  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`

Install the libraries via pip if needed:

```bash
pip install pandas numpy matplotlib seaborn
