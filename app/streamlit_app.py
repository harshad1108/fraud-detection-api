import streamlit as st
import requests

st.title("💳 Fraud Detection System")

features = []

for i in range(30):
    value = st.number_input(f"Feature V{i}", value=0.0)
    features.append(value)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": features}
    )

    result = response.json()

    st.write("Prediction:", result["prediction"])
    st.write("Fraud Probability:", result["fraud_probability"])
