import pandas as pd
from sqlalchemy import create_engine

# Load predicted file
df = pd.read_csv(
    "data/final_predictions.csv"
)

# Keep useful columns
result = df[
    [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "Prediction",
        "Churn_Probability"
    ]
]

# Connect DB
engine = create_engine(
    "postgresql://postgres:%40Supriya8116@localhost:5432/telecom_db"
)

# Save
result.to_sql(
    "customer_predictions",
    engine,
    if_exists="replace",
    index=False
)

print(
    "Saved to PostgreSQL"
)