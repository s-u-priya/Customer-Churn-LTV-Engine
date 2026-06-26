import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv(
    "data/processed_telco.csv"
)

# Encode target
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Features + Target
X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]

# Detect columns
categorical_cols = (
    X.select_dtypes(
        include="object"
    ).columns
)

numeric_cols = (
    X.select_dtypes(
        exclude="object"
    ).columns
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)

# Full pipeline
pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        LogisticRegression()
    )
])

# Split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# Train
pipeline.fit(
    X_train,
    y_train
)

# Save
joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)

print(
    "Pipeline Saved Successfully"
)
