import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel("data/telco_churn.xlsx")

# Remove unnecessary columns
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

# Convert Total Charges
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

df = df.dropna()

# Feature Engineering
df["Revenue_Per_Month"] = (
    df["Total Charges"] /
    (df["Tenure Months"] + 1)
)

df["Charge_Ratio"] = (
    df["Total Charges"] /
    (df["Monthly Charges"] + 1)
)

df["Support_Services"] = (
    (df["Online Security"] == "Yes").astype(int)
    +
    (df["Tech Support"] == "Yes").astype(int)
)

df["Streaming_Count"] = (
    (df["Streaming TV"] == "Yes").astype(int)
    +
    (df["Streaming Movies"] == "Yes").astype(int)
)

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Target = CLTV
X = df.drop("CLTV", axis=1)
y = df["CLTV"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))