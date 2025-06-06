import streamlit as st
import pandas as pd
import pickle as pk
import os

# Load model and scaler safely
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    model = pk.load(open('model.pkl', 'rb'))
    scaler = pk.load(open('scaler.pkl', 'rb'))
else:
    st.error("❌ 'model.pkl' or 'scaler.pkl' not found. Please upload them via 'Manage app' > 'Files'.")
    st.stop()

# App Header
st.header('Loan Eligibility Prediction Model')

# Input section
st.subheader("Enter Applicant Details")

num_of_dep = st.slider('Choose Number of Dependents', 0, 10)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed', ['Yes', 'No'])

annual_income = st.number_input('Enter Annual Income', min_value=0, step=1000)
loan_amt = st.number_input('Enter Loan Amount', min_value=0, step=1000)
loan_duration = st.slider('Select Loan Duration (Years)', 0, 20)
cibil_score = st.slider('Select CIBIL Score', 0, 900)
assets = st.number_input('Enter Value of Assets', min_value=0, step=1000)

# Encode categorical features
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 1 if self_emp == 'Yes' else 0

# Prediction Button
if st.button('Predict'):
    input_data = pd.DataFrame([[num_of_dep, grad_s, emp_s, annual_income, loan_amt,
                                 loan_duration, cibil_score, assets]], 
                               columns=['no_of_dependents', 'education', 'self_employed', 
                                        'income_annum', 'loan_amount', 'loan_term', 
                                        'cibil_score', 'Assets'])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make a prediction
    predict = model.predict(scaled_input)
    
    # Display the result
    if predict[0] == 1:
        st.success('✅ Loan Approved')
    else:
        st.error('❌ Loan Not Approved')