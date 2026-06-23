import streamlit as st
from Fraud_Detector import MobileDataGenerator, FeatureEngineer, FraudClassifier
import pandas as pd


# Initialize model
gen = MobileDataGenerator()
fe = FeatureEngineer()
model = FraudClassifier()

# Train model once
df = gen.generate()
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
X = fe.transform(X)
model.fit(X, y)


st.title("💳 Mobile Fraud Detection AI App")

st.write("Enter transaction details below:")

transaction_amount = st.number_input("Transaction Amount", value=1000)
transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
transactions_per_day = st.number_input("Transactions per Day", value=5)
avg_transaction_amount = st.number_input("Average Transaction Amount", value=800)
device_age_days = st.number_input("Device Age (days)", value=100)
new_device = st.selectbox("New Device?", [0, 1])
location_change = st.selectbox("Location Change?", [0, 1])
distance_from_home_km = st.number_input("Distance from Home (km)", value=10.0)
vpn_usage = st.selectbox("VPN Usage?", [0, 1])
session_duration_sec = st.number_input("Session Duration (sec)", value=300)
login_attempts = st.number_input("Login Attempts", value=1)
failed_logins = st.number_input("Failed Logins", value=0)
typing_speed_wpm = st.number_input("Typing Speed", value=50)
api_calls_per_minute = st.number_input("API Calls per min", value=5)
network_type = st.selectbox("Network Type", ["wifi", "4g", "5g", "3g"])
account_age_days = st.number_input("Account Age (days)", value=365)


if st.button("Predict Fraud Risk"):
    data = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "transaction_hour": transaction_hour,
        "transactions_per_day": transactions_per_day,
        "avg_transaction_amount": avg_transaction_amount,
        "device_age_days": device_age_days,
        "new_device": new_device,
        "location_change": location_change,
        "distance_from_home_km": distance_from_home_km,
        "vpn_usage": vpn_usage,
        "session_duration_sec": session_duration_sec,
        "login_attempts": login_attempts,
        "failed_logins": failed_logins,
        "typing_speed_wpm": typing_speed_wpm,
        "api_calls_per_minute": api_calls_per_minute,
        "network_type": network_type,
        "account_age_days": account_age_days
    }])

    data = fe.transform(data)
    prob = model.predict_proba(data)[0]

    st.success(f"Fraud Probability: {round(prob*100,2)}%")

    if prob > 0.7:
        st.error("HIGH RISK 🚨 Fraud Detected")
    elif prob > 0.3:
        st.warning("MEDIUM RISK ⚠")
    else:
        st.success("LOW RISK ✅ Safe Transaction")