import joblib
import pandas as pd

# Load model
model = joblib.load("models/churn_model.pkl")

# Example customer
sample_customer = pd.DataFrame([{
    "Gender": 0,
    "Senior Citizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "Tenure Months": 12,
    "Phone Service": 1,
    "Multiple Lines": 0,
    "Internet Service": 1,
    "Online Security": 0,
    "Online Backup": 1,
    "Device Protection": 0,
    "Tech Support": 0,
    "Streaming TV": 1,
    "Streaming Movies": 1,
    "Contract": 0,
    "Paperless Billing": 1,
    "Payment Method": 2,
    "Monthly Charges": 75.50,
    "Total Charges": 906.00
}])

prediction = model.predict(sample_customer)

probability = model.predict_proba(sample_customer)

print("Prediction:", prediction[0])
print("Churn Probability:", probability[0][1])