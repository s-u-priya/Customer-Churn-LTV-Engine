# Customer Churn Prediction & LTV Engine

## Project Overview

Customer Churn Prediction & Lifetime Value (LTV) Engine is an end-to-end machine learning project designed for telecommunications and subscription-based businesses.

The system predicts whether a customer is likely to churn and calculates churn probability to help businesses identify customers requiring retention efforts.

Predictions are served through FastAPI, stored in PostgreSQL, and visualized using Metabase dashboards.

---

## Features

* Customer churn prediction using Machine Learning
* Churn probability scoring
* Batch prediction for all customers
* FastAPI REST API deployment
* PostgreSQL prediction storage
* Metabase dashboard visualization
* Customer segmentation for high-risk users

---

## Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression

### Backend

* FastAPI
* Uvicorn

### Database

* PostgreSQL
* SQLAlchemy

### Dashboard

* Metabase

---

## Project Architecture

Dataset
↓
Data Cleaning & EDA
↓
Feature Engineering
↓
Model Training
↓
Pipeline Serialization (.pkl)
↓
FastAPI Deployment
↓
PostgreSQL Storage
↓
Metabase Dashboard

---

## Project Structure

```plaintext
Customer_Churn_Project/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed_telco.csv
│   └── final_predictions.csv
│
├── models/
│   ├── churn_model.pkl
│   └── churn_pipeline.pkl
│
├── notebooks/
│   ├── EDA.py
│   ├── Modeling.py
│   ├── Test_Model.py
│   ├── batch_predict.py
│   └── save_predictions.py
│
├── dashboard/
│   └── metabase.jar
│
├── screenshots/
│
├── requirements.txt
├── README.md
```

---

## Model Performance

### Logistic Regression

Accuracy: **82.11%**

Classification Report:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.69  |
| Recall    | 0.60  |
| F1 Score  | 0.64  |

---

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Customer Churn API Running"
}
```

---

### Predict Customer Churn

```http
POST /predict_churn
```

Example Request:

```json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":2,
  "PhoneService":"Yes",
  "MultipleLines":"No",
  "InternetService":"Fiber optic",
  "OnlineSecurity":"No",
  "OnlineBackup":"No",
  "DeviceProtection":"No",
  "TechSupport":"No",
  "StreamingTV":"Yes",
  "StreamingMovies":"Yes",
  "Contract":"Month-to-month",
  "PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check",
  "MonthlyCharges":120,
  "TotalCharges":240,
  "Revenue":240
}
```

Example Response:

```json
{
  "prediction":"CHURN",
  "probability":73.08
}
```

---

## How To Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
python -m uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### Run Batch Prediction

```bash
python notebooks/batch_predict.py
```

---

### Save Predictions To PostgreSQL

```bash
python notebooks/save_predictions.py
```

---

### Run Dashboard

```bash
cd dashboard
java -jar metabase.jar
```

Open:

```text
http://localhost:3000
```

---

## Dashboard Metrics

* Total Predictions
* Average Churn Risk
* Prediction Distribution
* Average Monthly Charges
* High Risk Customers

---

## Future Improvements

* Customer Lifetime Value (LTV)
* Docker deployment
* Cloud deployment
* Real-time prediction pipeline
* Automated retraining

---

## Author

Supriya
