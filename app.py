# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model (Make sure this .pkl is in the same folder)
try:
    model = joblib.load("fraud_detection_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'fraud_detection_model.pkl' is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict the probability of fraud.")

# Input form
with st.form("fraud_form"):
    V_features = []
    for i in range(1, 29):
        V_input = st.number_input(f"V{i}", value=0.0)
        V_features.append(V_input)

    Time = st.number_input("Transaction Time", value=0.0)
    Amount = st.number_input("Transaction Amount", value=0.0)

    submitted = st.form_submit_button("Predict")

# On submit
if submitted:
    # Arrange features in expected order: V1–V28, Time, Amount
    features = np.array(V_features + [Time, Amount]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Display result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected with {probability * 100:.2f}% confidence!")
    else:
        st.success(f"✅ Transaction is Normal with {100 - probability * 100:.2f}% confidence.")
