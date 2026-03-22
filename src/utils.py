import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_object(file_path, obj):
    """Save Python object as pickle file"""
    try:
        # Create directory if doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Saving object to: {file_path}")
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"✅ Object saved successfully: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving object: {e}")
        raise Exception(f"Error saving object: {e}")

def load_object(file_path):
    """Load Python object from pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.error(f"❌ Error loading object: {e}")
        raise Exception(f"Error loading object: {e}")

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluate multiple models with GridSearchCV hyperparameter tuning"""
    try:
        report = {}
        
        for model_name in models.keys():
            logger.info(f"🔄 Training {model_name}...")
            
            model = models[model_name]
            param_grid = params[model_name]
            
            # GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Test accuracy with best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            report[model_name] = test_accuracy
            logger.info(f"✅ {model_name}: {test_accuracy:.4f} | Best params: {grid_search.best_params_}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        raise Exception(f"Model evaluation failed: {e}")
