import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load the trained model
model = tf.keras.models.load_model('customer_churn_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=95, value=25)
tenure = st.number_input("Tenure (in years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict"):
    try:
        # Create input dataframe with the same structure as training data
        input_data = pd.DataFrame({
            "CreditScore": [600],  # Default credit score since it's not in input
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # Encode gender
        input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])
        
        # One-hot encode geography
        geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
        geography_feature_names = onehot_encoder_geography.get_feature_names_out(['Geography'])
        
        # Create geography dataframe
        geography_df = pd.DataFrame(geography_encoded, columns=geography_feature_names)
        
        # Rename geography columns to match training data
        geography_df = geography_df.rename(columns={
            'Geography_France': 'Geography_France',
            'Geography_Germany': 'Geography_Germany',
            'Geography_Spain': 'Geography_Spain'
        })
        
        # Combine the dataframes
        input_data = pd.concat([input_data, geography_df], axis=1)
        
        # Ensure all features are in the correct order (same as training)
        expected_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                           'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                           'Geography_France', 'Geography_Germany', 'Geography_Spain']
        
        # Reorder columns to match training data
        input_data = input_data[expected_features]
        
        # Scale the features
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display results
        st.subheader("Prediction Results")
        if prediction_proba > 0.5:
            st.error(f"⚠️ The customer is likely to CHURN (Leave)")
            st.write(f"Churn Probability: {prediction_proba:.2%}")
        else:
            st.success(f"✅ The customer is likely to STAY")
            st.write(f"Churn Probability: {prediction_proba:.2%}")
            
        # Show input summary
        st.subheader("Input Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Geography:** {geography}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Age:** {age}")
            st.write(f"**Tenure:** {tenure} years")
        with col2:
            st.write(f"**Balance:** ${balance:,.2f}")
            st.write(f"**Products:** {num_of_products}")
            st.write(f"**Credit Card:** {'Yes' if has_cr_card else 'No'}")
            st.write(f"**Active Member:** {'Yes' if is_active_member else 'No'}")
            st.write(f"**Estimated Salary:** ${estimated_salary:,.2f}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Developed by Anujan</strong></p>
        <p>Faculty of IT, University of Moratuwa</p>
    </div>
    """, 
    unsafe_allow_html=True
)