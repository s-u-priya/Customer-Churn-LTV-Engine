import joblib

model = joblib.load(
    "models/churn_model.pkl"
)

print(
    type(model)
)

print(
    "Model Loaded Successfully"
)