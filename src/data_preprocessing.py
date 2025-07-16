import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """Load and return the dataset"""
        df = pd.read_csv(file_path)
        # Convert Time to seconds since start of day
        df['Time'] = df['Time'] % 86400
        return df

    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Separate features and target
        X = df.drop('Class', axis=1).values
        y = df['Class'].values

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def split_data(self, X, y, test_size=0.2):
        """Split data into training and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Convert to numpy arrays for compatibility
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        return X_train, X_test, y_train, y_test

    def save_scaler(self, file_path):
        """Save the fitted scaler for future use"""
        joblib.dump(self.scaler, file_path)

    def load_scaler(self, file_path):
        """Load a previously saved scaler"""
        self.scaler = joblib.load(file_path)

    def preprocess_input(self, transaction_data):
        """Preprocess a single transaction for prediction"""
        # Convert to DataFrame if not already
        if not isinstance(transaction_data, pd.DataFrame):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Scale the data using the saved scaler
        self.load_scaler('models/scaler.pkl')
        X_scaled = self.scaler.transform(transaction_data)
        
        return X_scaled
