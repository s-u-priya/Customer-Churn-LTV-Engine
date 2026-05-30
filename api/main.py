from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

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


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API"
    }


@app.post("/predict")
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

    probability = float(
        model.predict_proba(data)[0][1]
    )

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4)
    }