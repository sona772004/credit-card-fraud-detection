# Credit Card Fraud Detection System

An end-to-end machine learning project that detects fraudulent credit card transactions using advanced AI/ML techniques.

## Project Overview

This project implements a credit card fraud detection system that:
- Processes and analyzes transaction data
- Trains machine learning models to detect fraudulent patterns
- Provides a user-friendly web interface for real-time predictions
- Implements various ML algorithms for comparison

## Features

- Data preprocessing and feature engineering
- Logistic Regression model for fraud detection
- Real-time prediction interface using Streamlit
- Model performance visualization
- Feature scaling and normalization
- Model persistence and loading

## Testing the Application

1. Run the training script first:
```bash
python train_models.py
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Test the application:
   - Open your web browser and go to the URL shown in the terminal (typically http://localhost:8501)
   - Enter transaction details in the input fields:
     - Time: Enter transaction time in seconds (e.g., 1000 for a transaction at 16:40)
     - Amount: Enter transaction amount
     - V1-V28: Enter values between -1 and 1 (these are normalized features)
   - Click "Predict Fraud" to see the prediction
   - The app will show whether the transaction is likely fraudulent or legitimate
   - A bar chart of feature values will be displayed for analysis

## Setup Instructions

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

- `data/`: Contains dataset and processed data
- `models/`: Trained ML models
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Source code for the application
  - `data_preprocessing.py`: Data cleaning and preprocessing
  - `model_training.py`: ML model training
  - `prediction.py`: Prediction logic
  - `app.py`: Streamlit web application

## Usage

1. Train models using the Jupyter notebook
2. Use the web interface to make predictions on new transactions
3. View model performance metrics and visualizations

## Technologies Used

- Python
- TensorFlow/Keras
- Scikit-learn
- Streamlit
- Pandas/Numpy
