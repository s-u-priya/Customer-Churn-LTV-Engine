import pandas as pd
import joblib

# Load data
df = pd.read_csv(
    "data/processed_telco.csv"
)

# Remove target/customer id
X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ],
    errors="ignore"
)

# Load pipeline
pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

# Predict
df["Prediction"] = (
    pipeline.predict(X)
)

# Probability
df["Churn_Probability"] = (
    pipeline
    .predict_proba(X)[:, 1]
    * 100
)

# Convert labels
df["Prediction"] = (
    df["Prediction"]
    .map({
        1: "CHURN",
        0: "STAY"
    })
)

# Save
df.to_csv(
    "data/final_predictions.csv",
    index=False
)

print(
    "Prediction Complete"
)

print(
    df[
        [
            "customerID",
            "Prediction",
            "Churn_Probability"
        ]
    ].head()
)