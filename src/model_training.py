from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
class ModelTrainer:
    def __init__(self):
        self.models = {}

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        # Use a smaller subset of the data
        subset_size = 10000  # Use only 10,000 samples
        np.random.seed(42)
        indices = np.random.choice(len(X_train), size=subset_size, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        # Simple parameter grid
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l1', 'l2']
        }
        
        lr = LogisticRegression(random_state=42, solver='liblinear')
        lr.fit(X_subset, y_subset)
        
        self.models['LogisticRegression'] = lr
        return {'C': lr.C, 'penalty': 'l1'}, lr.score(X_subset, y_subset)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        return report, cm

    def save_model(self, model_name, model, file_path):
        """Save trained model"""
        joblib.dump(model, file_path)

    def load_model(self, model_name, file_path):
        """Load a trained model"""
        return joblib.load(file_path)
