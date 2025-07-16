import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import joblib

class FraudPredictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.models = {}

    def load_models(self, model_paths):
        """Load trained models"""
        for model_name, path in model_paths.items():
            self.models[model_name] = self.trainer.load_model(model_name, path)

    def preprocess_input(self, transaction_data):
        """Preprocess input data for prediction"""
        # Convert input to DataFrame if not already
        if not isinstance(transaction_data, pd.DataFrame):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Scale the data using the saved scaler
        self.preprocessor.load_scaler('models/scaler.pkl')
        X_scaled = self.preprocessor.scaler.transform(transaction_data)
        
        return X_scaled

    def predict(self, transaction_data, model_name='RandomForest'):
        """Make fraud prediction using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        X_processed = self.preprocess_input(transaction_data)
        
        model = self.models[model_name]
        prediction = model.predict(X_processed)[0]
        
        return prediction
