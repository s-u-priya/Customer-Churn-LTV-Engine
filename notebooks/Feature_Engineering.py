import pandas as pd

df = pd.read_csv("data/telco.csv")

df["TotalCharges"] = (
    df["TotalCharges"]
    .replace(" ", "0")
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"]
)

df["Revenue"] = (
    df["MonthlyCharges"]
    * df["tenure"]
)

df.to_csv(
    "data/processed_telco.csv",
    index=False
)

print("Processed dataset saved")