
import streamlit as st
import joblib
from pathlib import Path

BASE = Path(__file__).parent

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return (
        joblib.load(BASE / "logistic_regression_model.pkl"),
        joblib.load(BASE / "scaler.pkl"),
        joblib.load(BASE / "feature_columns.pkl"),
    )

model, scaler, feature_columns = load_model()

st.title("📊 Customer Churn Prediction System")
st.write("Predict customer churn using Machine Learning.")
st.divider()
st.success("Machine Learning model loaded successfully!")
