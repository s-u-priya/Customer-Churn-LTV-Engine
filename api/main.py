from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    Predict customer churn risk using a machine learning model.

    Features:
    - Single Customer Prediction
    - Batch Prediction
    - Probability Scores
    - XGBoost Model
    """,
    version="1.0.0"
)

# Load model
model = joblib.load("models/churn_model.pkl")


class Customer(BaseModel):
    Gender: int
    Senior_Citizen: int
    Partner: int
    Dependents: int
    Tenure_Months: int
    Phone_Service: int
    Multiple_Lines: int
    Internet_Service: int
    Online_Security: int
    Online_Backup: int
    Device_Protection: int
    Tech_Support: int
    Streaming_TV: int
    Streaming_Movies: int
    Contract: int
    Paperless_Billing: int
    Payment_Method: int
    Monthly_Charges: float
    Total_Charges: float

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": 0,
                "Senior_Citizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "Tenure_Months": 12,
                "Phone_Service": 1,
                "Multiple_Lines": 0,
                "Internet_Service": 1,
                "Online_Security": 0,
                "Online_Backup": 1,
                "Device_Protection": 0,
                "Tech_Support": 0,
                "Streaming_TV": 1,
                "Streaming_Movies": 1,
                "Contract": 0,
                "Paperless_Billing": 1,
                "Payment_Method": 2,
                "Monthly_Charges": 75.5,
                "Total_Charges": 900.0
            }
        }


@app.get("/", tags=["General"])
def home():
    return {
        "project": "Customer Churn Prediction & LTV Engine",
        "model": "XGBoost",
        "accuracy": "81.45%",
        "version": "1.0.0",
        "status": "Running"
    }


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict", tags=["Predictions"])
def predict(customer: Customer):

    data = pd.DataFrame([{
        "Gender": customer.Gender,
        "Senior Citizen": customer.Senior_Citizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "Tenure Months": customer.Tenure_Months,
        "Phone Service": customer.Phone_Service,
        "Multiple Lines": customer.Multiple_Lines,
        "Internet Service": customer.Internet_Service,
        "Online Security": customer.Online_Security,
        "Online Backup": customer.Online_Backup,
        "Device Protection": customer.Device_Protection,
        "Tech Support": customer.Tech_Support,
        "Streaming TV": customer.Streaming_TV,
        "Streaming Movies": customer.Streaming_Movies,
        "Contract": customer.Contract,
        "Paperless Billing": customer.Paperless_Billing,
        "Payment Method": customer.Payment_Method,
        "Monthly Charges": customer.Monthly_Charges,
        "Total Charges": customer.Total_Charges
    }])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    risk = "Low"

    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.3:
        risk = "Medium"

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk
    }


@app.post("/predict_batch", tags=["Predictions"])
def predict_batch(customers: List[Customer]):

    rows = []

    for customer in customers:
        rows.append({
            "Gender": customer.Gender,
            "Senior Citizen": customer.Senior_Citizen,
            "Partner": customer.Partner,
            "Dependents": customer.Dependents,
            "Tenure Months": customer.Tenure_Months,
            "Phone Service": customer.Phone_Service,
            "Multiple Lines": customer.Multiple_Lines,
            "Internet Service": customer.Internet_Service,
            "Online Security": customer.Online_Security,
            "Online Backup": customer.Online_Backup,
            "Device Protection": customer.Device_Protection,
            "Tech Support": customer.Tech_Support,
            "Streaming TV": customer.Streaming_TV,
            "Streaming Movies": customer.Streaming_Movies,
            "Contract": customer.Contract,
            "Paperless Billing": customer.Paperless_Billing,
            "Payment Method": customer.Payment_Method,
            "Monthly Charges": customer.Monthly_Charges,
            "Total Charges": customer.Total_Charges
        })

    df = pd.DataFrame(rows)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = []

    for pred, prob in zip(predictions, probabilities):

        risk = "Low"

        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.3:
            risk = "Medium"

        results.append({
            "prediction": int(pred),
            "churn_probability": round(float(prob), 4),
            "risk_level": risk
        })

    return results
