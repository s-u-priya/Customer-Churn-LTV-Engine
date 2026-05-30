import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction & LTV Engine")
chart_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        80.95,
        80.03,
        81.45
    ]
})

st.subheader("Model Performance Comparison")
st.bar_chart(
    chart_data.set_index("Model")
)
st.markdown("""
Predict customer churn risk using a machine learning model trained on
telecommunications customer data.
""")

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider(
    "Tenure Months",
    1,
    72,
    12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    value=75.50
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    value=900.00
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "XGBoost")

with col2:
    st.metric("Accuracy", "81.45%")

with col3:
    st.metric("Dataset Size", "7043")

if st.button("Predict Churn Risk"):

    sample = pd.DataFrame([{
        "Gender": 0,
        "Senior Citizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "Tenure Months": tenure,
        "Phone Service": 1,
        "Multiple Lines": 0,
        "Internet Service": 1,
        "Online Security": 0,
        "Online Backup": 1,
        "Device Protection": 0,
        "Tech Support": 0,
        "Streaming TV": 1,
        "Streaming Movies": 1,
        "Contract": 0,
        "Paperless Billing": 1,
        "Payment Method": 2,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Results")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if probability < 0.3:
        st.success("🟢 Low Churn Risk")

    elif probability < 0.7:
        st.warning("🟡 Medium Churn Risk")

    else:
        st.error("🔴 High Churn Risk")

    st.write(
        "Prediction:",
        "Will Churn" if prediction == 1 else "Will Stay"
    )
