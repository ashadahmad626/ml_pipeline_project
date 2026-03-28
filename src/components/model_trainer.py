import os
import sys
import json
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils.utils import save_object, evaluate_models, get_best_model

# Optional dependencies — skip gracefully if not available on this platform
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    logger.warning(f"XGBoost not available: {e}")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM not available: {e}")


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    report_path: str = os.path.join("artifacts", "model_report.json")


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_jobs=-1),
}

if XGBOOST_AVAILABLE:
    MODELS["XGBoost"] = XGBClassifier(
        random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1
    )

if LIGHTGBM_AVAILABLE:
    MODELS["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0],
    },
    "Extra Trees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRIDS["XGBoost"] = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
    }

if LIGHTGBM_AVAILABLE:
    PARAM_GRIDS["LightGBM"] = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        logger.info("Model Training started.")
        try:
            report = evaluate_models(X_train, y_train, X_test, y_test, MODELS, PARAM_GRIDS)

            best_name, best_info = get_best_model(report)
            best_model = best_info["model"]

            if best_info["f1_score"] < 0.60:
                raise ValueError(f"No model achieved acceptable F1 score. Best: {best_info['f1_score']}")

            save_object(self.config.model_path, best_model)
            logger.info(f"Best model: {best_name} | F1: {best_info['f1_score']} | AUC: {best_info['roc_auc']}")

            # Save report (serialize model object out)
            serializable_report = {
                k: {kk: vv for kk, vv in v.items() if kk != "model"}
                for k, v in report.items()
            }
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
            with open(self.config.report_path, "w") as f:
                json.dump({
                    "best_model": best_name,
                    "models": serializable_report
                }, f, indent=2)

            return best_name, best_info, report

        except Exception as e:
            raise CustomException(e, sys)