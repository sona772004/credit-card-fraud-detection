import pandas as pd
import numpy as np
import os
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Initialize preprocessor and load data
        logging.info("Initializing data preprocessor...")
        preprocessor = DataPreprocessor()
        
        # Check if dataset exists
        if not os.path.exists('data/creditcard.csv'):
            raise FileNotFoundError("Credit card dataset not found. Please place 'creditcard.csv' in the 'data' directory.")
            
        df = preprocessor.load_data('data/creditcard.csv')
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Display class distribution
        logging.info("\nClass distribution:")
        logging.info(df['Class'].value_counts())
        
        # Preprocess the data
        logging.info("\nPreprocessing data...")
        X, y = preprocessor.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        logging.info(f"\nTraining samples: {len(X_train)}")
        logging.info(f"Test samples: {len(X_test)}")
        
        # Save the scaler
        logging.info("\nSaving scaler...")
        preprocessor.save_scaler('models/scaler.pkl')
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Train Logistic Regression model
        logging.info("\nTraining Logistic Regression model...")
        best_params, best_score = trainer.train_logistic_regression(X_train, y_train)
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        # Save Logistic Regression model
        logging.info("\nSaving Logistic Regression model...")
        trainer.save_model('LogisticRegression', trainer.models['LogisticRegression'], 'models/logistic_regression_model.pkl')
        
        # Evaluate models
        logging.info("\nEvaluating models...")
        for model_name, model in trainer.models.items():
            logging.info(f"\n{model_name} performance:")
            report, cm = trainer.evaluate_model(model, X_test, y_test)
            logging.info("Classification Report:\n" + report)
            logging.info(f"Confusion Matrix:\n{cm}")
            
            # Save evaluation metrics
            with open(f'models/{model_name}_metrics.txt', 'w') as f:
                f.write(f"Classification Report:\n{report}\n\nConfusion Matrix:\n{cm}")
        
        logging.info("\nTraining completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.info("Starting model training...")
    main()
