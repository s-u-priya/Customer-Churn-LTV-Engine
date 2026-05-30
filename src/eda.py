import pandas as pd

# Load dataset
df = pd.read_excel("data/telco_churn.xlsx")

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns)

print("\n" + "=" * 50)
print("DATA INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("CHURN DISTRIBUTION")
print("=" * 50)
print(df["Churn Value"].value_counts())