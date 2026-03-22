import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.logger import logger  # ✅ FIXED: logger (not logging)
from src.exception import CustomException  # ✅ FIXED: CustomException
from dataclasses import dataclass
from src.utils import save_object, evaluate_models  # ✅ FIXED: evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model_trainer", "model.pkl")  # ✅ FIXED: spelling + name

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):  # ✅ FIXED: spelling
        try:
            logger.info("Model training initiated")  # ✅ logger.info
            
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logger.info(f"Train dataset: {X_train.shape}, Test: {X_test.shape}")
            
            # Models with proper names  ✅ FIXED: "Logastic" → "Logistic Regression"
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            # Hyperparameter grids  ✅ FIXED: Logistic params
            params = {
                "Random Forest": {
                    "class_weight": ["balanced"],
                    'n_estimators': [20, 50],  # ✅ Reduced for speed
                    'max_depth': [8, 10],
                    'min_samples_split': [2, 5],
                },
                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ['gini', "entropy"],
                    "splitter": ['best', 'random'],
                    "max_depth": [3, 5, 8],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "Logistic Regression": {
                    "class_weight": ["balanced"],
                    'penalty': ['l2'],  # ✅ FIXED: saga needs more memory
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear']
                }
            }

            # Evaluate models with GridSearchCV
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test,
                models=models, 
                params=params
            )
            
            # Get best model  ✅ FIXED: Logic error
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logger.info(f"✅ Best Model: {best_model_name}")
            logger.info(f"✅ Best Accuracy: {best_model_score:.4f}")
            
            # Train best model on FULL training data
            best_model.fit(X_train, y_train)
            
            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Test accuracy
            test_pred = best_model.predict(X_test)
            from sklearn.metrics import accuracy_score
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Final report
            model_trainer_report = {
                "best_model_name": best_model_name,
                "best_model_accuracy": best_model_score,
                "test_accuracy": test_accuracy,
                "all_models": model_report
            }
            
            logger.info(f"🎉 Model training completed!")
            logger.info(f"Model saved: {self.model_trainer_config.trained_model_file_path}")
            
            return model_trainer_report
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise CustomException(e, sys)
