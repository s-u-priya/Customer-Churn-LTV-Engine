import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv(
    "data/processed_telco.csv"
)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]

X = pd.get_dummies(
    X,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)
joblib.dump(
    model,
    "models/churn_model.pkl"
)

print(
    "Model Saved"
)

predictions = model.predict(
    X_test
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            predictions
        ) * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        predictions
    )
)