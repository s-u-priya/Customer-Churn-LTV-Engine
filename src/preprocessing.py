import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    "City",
    "Zip Code",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

df = df.drop(columns=drop_cols)

# Convert Total Charges
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

print(df.head())

print("\nData Types:")
print(df.dtypes)