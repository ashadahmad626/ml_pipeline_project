import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from src.logger import logger
from src.exception import CustomException


def save_object(file_path: str, obj):
    """Save any Python object using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a pickled Python object."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param_grids: dict) -> dict:
    """
    Evaluate multiple models with GridSearchCV and return detailed report.
    """
    report = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            logger.info(f"Training model: {name}")
            params = param_grids.get(name, {})

            if params:
                gs = GridSearchCV(model, params, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1")

            report[name] = {
                "model": best_model,
                "best_params": best_params,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
                "cv_f1_mean": round(cv_scores.mean(), 4),
                "cv_f1_std": round(cv_scores.std(), 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
            }
            logger.info(f"{name} → F1: {report[name]['f1_score']}, AUC: {report[name]['roc_auc']}")

        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")
            continue

    return report


def get_best_model(report: dict):
    """Return the best model name and its info based on F1 score."""
    best_name = max(report, key=lambda k: report[k]["f1_score"])
    return best_name, report[best_name]