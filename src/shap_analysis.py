import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset
df = pd.read_excel("data/telco_churn.xlsx")

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
    "CLTV",
    "Churn Reason"
]

df = df.drop(columns=drop_cols)

# Convert Total Charges
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

df = df.dropna()

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(
    shap_values,
    X_test,
    show=True
)