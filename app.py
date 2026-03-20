import streamlit as st
import joblib
import pandas as pd

# 1. Load the saved model
model = joblib.load('churn_model.pkl')

st.title("🚀 Customer Retention Predictor")
st.write("Predicting if a customer will leave based on their spending habits.")

# 2. Input fields (Only the ones the model needs)
frequency = st.number_input("Frequency (Total number of purchases)", min_value=1, value=5)
monetary = st.number_input("Monetary (Total money spent)", min_value=1.0, value=500.0)

if st.button("Predict Churn Status"):
    # 3. FEATURE ENGINEERING (This must happen inside the app too!)
    avg_spend = monetary / frequency

    # 4. Create the dataframe with EXACT names and order used in fit()
    # Order: Frequency, Monetary, Avg_Spend
    features = pd.DataFrame([[frequency, monetary, avg_spend]], 
                            columns=['Frequency', 'Monetary', 'Avg_Spend'])
    
    # 5. Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # 6. Display Results
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! Probability of Churn: {probability:.2%}")
        st.write("**Strategy:** Customer is showing signs of leaving. Offer a re-engagement discount.")
    else:
        st.success(f"✅ Active Customer. Probability of Churn: {probability:.2%}")
        st.write("**Strategy:** High engagement! Upsell premium products or loyalty tiers.")