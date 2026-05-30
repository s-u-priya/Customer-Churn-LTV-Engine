import pandas as pd

# Load dataset
df = pd.read_excel("data/telco_churn.xlsx")

# Drop unnecessary columns
drop_cols = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "Churn Reason"
]

df = df.drop(columns=drop_cols)

# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Remove rows with missing values
df = df.dropna()

# Check missing values
print("Missing Values:")
print(df.isnull().sum().sum())

# =========================
# Feature Engineering
# =========================

# Revenue generated per month
df["Revenue_Per_Month"] = (
    df["Total Charges"] /
    (df["Tenure Months"] + 1)
)

# Total charges relative to monthly charges
df["Charge_Ratio"] = (
    df["Total Charges"] /
    (df["Monthly Charges"] + 1)
)

# Number of support services
df["Support_Services"] = (
    (df["Online Security"] == "Yes").astype(int)
    +
    (df["Tech Support"] == "Yes").astype(int)
)

# Number of streaming services
df["Streaming_Count"] = (
    (df["Streaming TV"] == "Yes").astype(int)
    +
    (df["Streaming Movies"] == "Yes").astype(int)
)

print("\nNew Features Preview:")
print(df[[
    "Revenue_Per_Month",
    "Charge_Ratio",
    "Support_Services",
    "Streaming_Count"
]].head())

print("\nNew Shape:")
print(df.shape)
