# 💳 Credit Card Default Prediction & Model Leaderboard

An end-to-end Machine Learning and Data Science project that predicts customer credit card default risks using financial and demographic data. This repository features a structured ML pipeline from Exploratory Data Analysis (EDA) and Automated Hyperparameter Tuning (via Optuna) to Interactive Dashboard Deployment using Streamlit.

---

## 📌 Project Overview

Predicting credit card default is a critical risk management task for financial institutions. This project aims to analyze historical customer data, identify key behavioral indicators of default, and build robust predictive models.

### Key Highlights:
* **Exploratory Data Analysis (EDA):** Comprehensive analysis of customer demographics, payment status histories, and bill amounts.
* **Modular Pipeline Architecture:** Clean separation of data processing, training, hyperparameter tuning, model comparison, and evaluation scripts.
* **Model Benchmark & Leaderboard:** Evaluates LightGBM, XGBoost, Random Forest, and K-Nearest Neighbors (KNN).
* **Automated Tuning:** Uses **Optuna** with Cross-Validation to optimize hyperparameters for peak performance.
* **Interactive Web App:** A **Streamlit** dashboard visualizing model performance metrics, feature importances, confusion matrices, and offering real-time prediction inputs.

---

## 📂 Repository Structure

```text
final_project/
│
├── data/
│   └── UCI_Credit_Card.csv         # Raw Dataset
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis Notebook
│
├── models/
│   ├── best_model.pkl              # Optuna Tuned Best Model
│   ├── lightgbm.pkl                # Trained LightGBM Model
│   ├── xgboost.pkl                 # Trained XGBoost Model
│   ├── random_forest.pkl           # Trained Random Forest Model
│   └── knn.pkl                     # Trained KNN Model
│
├── outputs/
│   ├── model_comparison.csv        # Summary metrics table
│   ├── model_comparison_chart.png  # Leaderboard ROC-AUC Comparison Chart
│   ├── confusion_matrix.png        # Best Model Confusion Matrix
│   └── feature_importance.png      # Top 10 Feature Importances Plot
│
├── src/
│   ├── preprocessing.py            # Data Cleaning & Preprocessing Pipeline
│   ├── train.py                    # Base Models Training Script
│   ├── tune.py                     # Optuna Hyperparameter Optimization Script
│   ├── compare.py                  # Evaluation & Leaderboard Generator
│   └── evaluate.py                 # Best Model Visualizations Generator
│
├── app/
│   └── streamlit_app.py            # Streamlit Interactive Dashboard
│
├── preprocessor.pkl                # Saved Data Preprocessor Object
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation