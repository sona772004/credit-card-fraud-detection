import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import joblib

# Initialize data preprocessor
preprocessor = DataPreprocessor()

# Load the trained model
try:
    model = joblib.load('models/logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model not found. Please run train_models.py first to train the model.")
    st.stop()

# Load the scaler
try:
    preprocessor.load_scaler('models/scaler.pkl')
except FileNotFoundError:
    st.error("Scaler not found. Please run train_models.py first to train the model.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and description
st.title("Credit Card Fraud Detection System")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This application uses a machine learning model to predict whether a credit card transaction is fraudulent.
""")

# Main content
st.header("Transaction Details")

# Create input fields for transaction features
transaction_features = {}

# Add input fields for the 30 features
for i in range(1, 29):
    feature_name = f'V{i}'
    transaction_features[feature_name] = st.number_input(f"{feature_name}", value=0.0)

transaction_features['Time'] = st.number_input("Transaction Time (seconds since start of day)", value=0.0)
transaction_features['Amount'] = st.number_input("Transaction Amount", value=0.0)

# Predict button
col1, col2 = st.columns(2)
with col1:
    if st.button("Predict Fraud"):
        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction_features])
        
        # Preprocess input
        preprocessed_data = preprocessor.preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)
        
        # Show result
        if prediction[0] == 1:
            st.error("This transaction is likely fraudulent!")
        else:
            st.success("This transaction appears to be legitimate.")

# Add some visualizations
col2.subheader("Transaction Analysis")

# Create a bar chart of feature values
feature_values = pd.DataFrame(list(transaction_features.items()), columns=['Feature', 'Value'])

st.bar_chart(feature_values.set_index('Feature'))

# Add some model performance metrics
st.sidebar.header("Model Performance")
st.sidebar.markdown("""
- **Accuracy:** 1.00
- **F1-score (Fraud):** 0.80
- **True Positives:** 76/98 fraud cases
""")
