from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import joblib

app = FastAPI()

# Load pipeline
pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

# PostgreSQL connection
engine = create_engine(
    "postgresql://postgres:%40Supriya8116@localhost:5432/telecom_db"
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    Revenue: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn API Running"
    }


@app.post("/predict_churn")
def predict(customer: Customer):

    df = pd.DataFrame([
        customer.model_dump()
    ])

    prediction = pipeline.predict(
        df
    )[0]

    probability = (
        pipeline
        .predict_proba(df)[0][1]
    )

    result = pd.DataFrame([{

"customerID":
f"CUST_{customer.tenure}_{int(customer.MonthlyCharges)}",

"tenure":
customer.tenure,

"MonthlyCharges":
customer.MonthlyCharges,

"Prediction":
(
"CHURN"
if prediction == 1
else "STAY"
),

"Churn_Probability":
round(
probability*100,
2
),

"Predicted_LTV":
round(
customer.MonthlyCharges *
customer.tenure *
0.85,
2
)

}])

    result.to_sql(
        "customer_predictions",
        engine,
        if_exists="append",
        index=False
    )

    return {
        "prediction":
            "CHURN"
            if prediction == 1
            else "STAY",

        "probability":
            round(
                probability * 100,
                2
            )
    }
