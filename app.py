import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer information to predict the likelihood of churn.")

st.divider()

# Customer Information
st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    partner = st.selectbox(
        "Has Partner?",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

st.divider()

# Internet Services
st.subheader("🌐 Internet Services")

col3, col4 = st.columns(2)

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col4:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.divider()

# Contract and Payment
st.subheader("💳 Contract and Payment")

col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col6:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()

st.info("Customer information form created successfully!")
