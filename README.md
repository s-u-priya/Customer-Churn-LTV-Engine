# Customer Churn Prediction & Lifetime Value (LTV) Engine

## Overview

A machine learning-powered analytics system designed to predict customer churn and estimate Customer Lifetime Value (CLTV) for subscription-based businesses.

The project helps organizations identify customers at risk of leaving and prioritize retention strategies based on customer value.

---

## Features

- Customer Churn Prediction
- Customer Lifetime Value (CLTV) Analysis
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression Model
- Random Forest Model
- XGBoost Model
- Feature Importance Analysis
- SHAP Explainability
- FastAPI REST API
- Batch Prediction Endpoint
- Streamlit Dashboard

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

**Records:** 7,043 customers

### Key Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Contract Type
- Internet Service
- Monthly Charges
- Total Charges
- Churn Status
- CLTV

---

## Tech Stack

### Languages

- Python
- SQL

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Joblib

### Backend

- FastAPI

### Visualization

- Streamlit

### Version Control

- Git
- GitHub

---

## Machine Learning Models

| Model | Accuracy |
|---------|----------|
| Logistic Regression | 80.95% |
| Random Forest | 80.03% |
| XGBoost | 81.45% |

### Best Model

**XGBoost**

Accuracy: **81.45%**

---

## Project Structure

```text
Customer-Churn-LTV-Engine
│
├── api
│   └── main.py
│
├── dashboard
│   └── app.py
│
├── data
│   └── telco_churn.xlsx
│
├── models
│   └── churn_model.pkl
│
├── src
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── feature_importance.py
│   ├── shap_analysis.py
│   ├── ltv_model.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## API Endpoints

### General

- GET /

Returns API information.

- GET /health

Health check endpoint.

### Predictions

- POST /predict

Predict churn for a single customer.

- POST /predict_batch

Predict churn for multiple customers.

---

## Dashboard

The Streamlit dashboard allows users to:

- Enter customer information
- Predict churn probability
- View churn risk level
- Compare model performance

---

## Explainability

SHAP was used to explain model predictions and identify the most influential features affecting churn.

---

## Business Impact

- Reduce customer acquisition costs
- Improve customer retention
- Prioritize high-value customers
- Optimize marketing campaigns
- Support data-driven decision making

---

## Future Improvements

- PostgreSQL Data Warehouse
- Docker Deployment
- Apache Superset Dashboard
- Automated Model Retraining
- Cloud Deployment (AWS/Azure)

---

## Author

Supriya

Data Analytics & Machine Learning Project
