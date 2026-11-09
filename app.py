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

# Prepare customer data for prediction

customer_data = {column: 0 for column in feature_columns}

# Numerical features
customer_data["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
customer_data["tenure"] = tenure
customer_data["MonthlyCharges"] = monthly_charges
customer_data["TotalCharges"] = total_charges

# Categorical features
customer_data["gender_Male"] = 1 if gender == "Male" else 0
customer_data["Partner_Yes"] = 1 if partner == "Yes" else 0
customer_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0
customer_data["PhoneService_Yes"] = 1 if phone_service == "Yes" else 0

customer_data["MultipleLines_No phone service"] = (
    1 if multiple_lines == "No phone service" else 0
)
customer_data["MultipleLines_Yes"] = 1 if multiple_lines == "Yes" else 0

customer_data["InternetService_Fiber optic"] = (
    1 if internet_service == "Fiber optic" else 0
)
customer_data["InternetService_No"] = 1 if internet_service == "No" else 0

customer_data["OnlineSecurity_No internet service"] = (
    1 if online_security == "No internet service" else 0
)
customer_data["OnlineSecurity_Yes"] = 1 if online_security == "Yes" else 0

customer_data["OnlineBackup_No internet service"] = (
    1 if online_backup == "No internet service" else 0
)
customer_data["OnlineBackup_Yes"] = 1 if online_backup == "Yes" else 0

customer_data["DeviceProtection_No internet service"] = (
    1 if device_protection == "No internet service" else 0
)
customer_data["DeviceProtection_Yes"] = 1 if device_protection == "Yes" else 0

customer_data["TechSupport_No internet service"] = (
    1 if tech_support == "No internet service" else 0
)
customer_data["TechSupport_Yes"] = 1 if tech_support == "Yes" else 0

customer_data["StreamingTV_No internet service"] = (
    1 if streaming_tv == "No internet service" else 0
)
customer_data["StreamingTV_Yes"] = 1 if streaming_tv == "Yes" else 0

customer_data["StreamingMovies_No internet service"] = (
    1 if streaming_movies == "No internet service" else 0
)
customer_data["StreamingMovies_Yes"] = 1 if streaming_movies == "Yes" else 0

customer_data["Contract_One year"] = 1 if contract == "One year" else 0
customer_data["Contract_Two year"] = 1 if contract == "Two year" else 0

customer_data["PaperlessBilling_Yes"] = (
    1 if paperless_billing == "Yes" else 0
)

customer_data["PaymentMethod_Credit card (automatic)"] = (
    1 if payment_method == "Credit card (automatic)" else 0
)
customer_data["PaymentMethod_Electronic check"] = (
    1 if payment_method == "Electronic check" else 0
)
customer_data["PaymentMethod_Mailed check"] = (
    1 if payment_method == "Mailed check" else 0
)

# Convert dictionary into DataFrame
input_data = pd.DataFrame(
    [customer_data],
    columns=feature_columns
)

# Display prepared features
st.subheader("🔍 Prepared Customer Data")
st.write(input_data)
st.success(f"Successfully prepared {input_data.shape[1]} features!")

# Prediction button
if st.button("🔮 Predict Customer Churn", use_container_width=True):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("📊 Prediction Results")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.2f}%"
    )
