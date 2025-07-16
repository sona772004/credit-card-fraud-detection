import streamlit as st
import pandas as pd
import numpy as np
from src.prediction import FraudPredictor
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize predictor
predictor = FraudPredictor()

# Load models
model_paths = {
    'RandomForest': 'models/random_forest_model.pkl',
    'NeuralNetwork': 'models/neural_network_model.h5'
}
predictor.load_models(model_paths)

# Streamlit app
st.title("Credit Card Fraud Detection System")

# Sidebar
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["LogisticRegression", "RandomForest", "GradientBoosting"]
)

# Main content
st.header("Transaction Analysis")

# Transaction input form
with st.form("transaction_form"):
    st.write("Enter transaction details:")
    
    # Create input fields for transaction features
    # Note: In a real application, you would have specific fields based on your dataset
    transaction_time = st.number_input("Transaction Time (in seconds)", min_value=0)
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    merchant_category = st.selectbox("Merchant Category", ["Retail", "Online", "Food", "Services"])
    
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    # Create sample transaction data (in real app, this would be based on actual dataset features)
    sample_data = {
        "Time": [transaction_time],
        "Amount": [amount],
        "V1": [np.random.normal()],  # Example feature
        "V2": [np.random.normal()],  # Example feature
        "V3": [np.random.normal()],  # Example feature
        # Add more features as needed based on your dataset
    }
    
    transaction_df = pd.DataFrame(sample_data)
    
    # Make prediction
    prediction = predictor.predict(transaction_df, selected_model)
    
    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Transaction Approved")

# Add visualizations
st.header("Model Performance")

# Example performance metrics (in real app, these would be based on actual model results)
performance_metrics = pd.DataFrame({
    "Model": ["RandomForest", "NeuralNetwork"],
    "Accuracy": [0.98, 0.97],
    "Precision": [0.95, 0.94],
    "Recall": [0.96, 0.95],
    "F1 Score": [0.95, 0.94]
})

st.dataframe(performance_metrics)

# Add some example visualizations
st.header("Transaction Analysis")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=performance_metrics, ax=ax)
plt.title("Model Performance Comparison")
st.pyplot(fig)
