# Customer Churn Prediction & LTV Engine

## Overview

A machine learning project that predicts customer churn for a telecom business and provides customer retention insights.

The system analyzes customer demographics, subscription details, billing information, and service usage patterns to identify customers likely to leave the company.

## Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Churn Prediction using Logistic Regression
* Random Forest comparison model
* Feature Importance Analysis
* Model Serialization using Joblib
* FastAPI-based Prediction Service

## Dataset

Telco Customer Churn Dataset

Dataset Size:

* 7043 customers
* 33 original features

Target Variable:

* Churn Value

  * 0 = Customer stays
  * 1 = Customer churns

## Technologies Used

### Programming

* Python

### Machine Learning

* Pandas
* NumPy
* Scikit-Learn

### API

* FastAPI
* Uvicorn

### Model Storage

* Joblib

## Project Structure

Customer-Churn-LTV-Engine/

├── api/

├── data/

├── models/

│   └── churn_model.pkl

├── src/

│   ├── eda.py

│   ├── preprocessing.py

│   ├── train_model.py

│   ├── random_forest.py

│   ├── feature_importance.py

│   ├── save_model.py

│   └── predict.py

├── requirements.txt

└── README.md

## Model Performance

### Logistic Regression

* Accuracy: 80.95%
* Churn Recall: 59%
* Churn F1 Score: 0.63

### Random Forest

* Accuracy: 80.03%
* Churn Recall: 51%
* Churn F1 Score: 0.59

## Top Churn Factors

1. Total Charges
2. Tenure Months
3. Monthly Charges
4. Contract Type
5. Online Security
6. Payment Method
7. Dependents
8. Tech Support

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run API:

python -m uvicorn api.main:app --reload

Open Swagger UI:

http://127.0.0.1:8000/docs

## Future Improvements

* XGBoost model
* SHAP explainability
* PostgreSQL integration
* Docker deployment
* Interactive dashboards using Superset or Metabase
