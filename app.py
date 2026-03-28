import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
model = joblib.load('churn_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🚀")
st.title("🚀 Customer Churn Predictor")

# 2. User Inputs (Capitalized variable names)
col1, col2 = st.columns(2)

with col1:
    # We name these 'Frequency' and 'Monetary' to match the model's needs
    Frequency = st.number_input("Total Purchases (Frequency)", min_value=1, value=5)
    Monetary = st.number_input("Total Spent ($)", min_value=1.0, value=500.0)
    Customer_Lifetime = st.number_input("Days since first purchase (Tenure)", min_value=0, value=100)

if st.button("Predict Churn Status"):
    # 3. BACKGROUND FEATURE ENGINEERING (All Capitalized)
    Avg_Spend = Monetary / Frequency
    Purchase_Interval = Customer_Lifetime / max((Frequency - 1), 1)
    Daily_Value = Monetary / (Customer_Lifetime + 1)
    Log_Monetary = np.log1p(Monetary)

    # 4. Create the input dictionary 
    # Keys and Values now match perfectly in case and name
    input_data = {
        'Frequency': Frequency,
        'Avg_Spend': Avg_Spend,
        'Customer_Lifetime': Customer_Lifetime,
        'Purchase_Interval': Purchase_Interval,
        'Daily_Value': Daily_Value,
        'Log_Monetary': Log_Monetary
    }
    
    # 5. Convert to DataFrame
    features = pd.DataFrame([input_data])
    
    # 6. Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # 7. Display Results
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk!** Churn Probability: {probability:.2%}")
    else:
        st.success(f"✅ **Low Risk.** Churn Probability: {probability:.2%}")