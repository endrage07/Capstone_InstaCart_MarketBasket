# Instacart Market Basket ML Capstone

## Overview
This project shows and ML workflow using Instacart Market Basket Analysis dataset. It covers problem framing, data preprocessing, feature engineering, model training, evaluation, and explainability.

## Dataset
- Source: Instacart Market Basket Analysis (Kaggle)
- Access method: KaggleHub (loaded dynamically in Google Colab)
- Core tables used:
  - orders
  - order_products__prior
  - order_products__train
  - products
  - aisles
  - departments

Raw dataset files are not committed to this repository due to size constraints.

## Repository Structure
- `notebooks/` – EDA, feature engineering, modeling, and explainability
- `src/` – Reproducible pipeline scripts

## Modeling Approach

- Problem type: Binary classification (reordered vs not reordered)
- Baseline model: Logistic Regression
- Primary model: Random Forest Classifier
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC


## Reproducibility
The notebooks are designed to be run sequentially. Instructions for fully reproducible execution are provided in later stages of the project.
