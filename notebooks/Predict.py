import pandas as pd
import joblib

# Load saved pipeline
pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

# New customer data
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 120,
    "TotalCharges": 240,
    "Revenue": 240
}])
# Predict class
prediction = pipeline.predict(customer)

# Predict probability
probability = pipeline.predict_proba(customer)

churn_probability = probability[0][1]
stay_probability = probability[0][0]

print("\nRESULT")

print(
    "Prediction:",
    "CHURN"
    if prediction[0] == 1
    else "STAY"
)

print(
    "Churn Probability:",
    round(
        churn_probability * 100,
        2
    ),
    "%"
)

print(
    "Stay Probability:",
    round(
        stay_probability * 100,
        2
    ),
    "%"
)