import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the "Honest" model
# Make sure the filename matches your final saved model
model = joblib.load('churn_model.pkl')

st.title("🚀 Customer Retention Predictor")
st.write("Predicting churn based on behavioral patterns and tenure.")

# 2. User Inputs (The 'Raw' Data)
col1, col2 = st.columns(2)

with col1:
    frequency = st.number_input("Total Purchases (Frequency)", min_value=1, value=5)
    monetary = st.number_input("Total Spent ($)", min_value=1.0, value=500.0)
    customer_lifetime = st.number_input("Days since first purchase (Tenure in Days)", min_value=0, value=100)

if st.button("Predict Churn Status"):
    # 3. BACKGROUND FEATURE ENGINEERING 
    # Must match the training logic exactly
    avg_spend = monetary / frequency
    purchase_interval = customer_lifetime / max((frequency - 1), 1)
    daily_value = monetary / (customer_lifetime + 1)
    log_monetary = np.log1p(monetary)

    # 4. Create the input dictionary to ensure order is handled by DataFrame
    input_data = {
        'Frequency': frequency,
        'Avg_Spend': avg_spend,
        'Customer_Lifetime': customer_lifetime,
        'Purchase_Interval': purchase_interval,
        'Daily_Value': daily_value,
        'Log_Monetary': log_monetary
    }
    
    # Convert to DataFrame
    features = pd.DataFrame([input_data])
    
    # 5. Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # 6. Display Results
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk!** Probability of Churn: {probability:.2%}")
        st.progress(probability)
        st.write("💡 **Action:** This customer is slowing down. Send a re-engagement discount.")
    else:
        st.success(f"✅ **Low Risk.** Probability of Churn: {probability:.2%}")
        st.progress(probability)
        st.write("💡 **Action:** High engagement! Keep them happy with loyalty points.")