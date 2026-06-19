import pandas as pd

df = pd.read_csv("data/telco.csv")

df["TotalCharges"] = (
    df["TotalCharges"]
    .replace(" ", "0")
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"]
)

charges_churn = df.groupby(
    "Churn"
)["MonthlyCharges"].mean()

print(
    charges_churn.round(2)
)
