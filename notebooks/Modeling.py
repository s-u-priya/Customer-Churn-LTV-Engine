import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/processed_telco.csv")

# Convert target to numbers
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Features and target
X = df.drop(
    columns=[
        "customerID",
        "Churn",
        "Revenue"
    ]
)

y = df["Churn"]

# Convert text columns to numeric
X = pd.get_dummies(
    X,
    drop_first=True
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Random Forest Model Trained")

# Predictions
predictions = model.predict(
    X_test
)

# Show first predictions
print("\nFirst 10 Predictions:")
print(predictions[:10])

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

#Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(
    importance.head(10)
)