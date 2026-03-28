import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.logger import logger
from src.exception import CustomException
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")


# ─── Column definitions ───────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country"
]
NUMERICAL_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]
DROP_COLS = []   # fnlwgt is noisy but kept for now; drop if desired
TARGET_COL = "income"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, strip whitespace, handle special chars."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Normalize target — handle both string ("<=50K"/">50K") and already-numeric (0/1)
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object:
            df[TARGET_COL] = df[TARGET_COL].str.strip().str.replace(".", "", regex=False)
            df[TARGET_COL] = df[TARGET_COL].map({"<=50K": 0, ">50K": 1})
        else:
            df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific engineered features."""
    df = df.copy()
    # Net capital
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]
    # Work intensity
    df["work_intensity"] = df["hours_per_week"] * df["age"]
    # High education flag
    df["high_education"] = (df["education_num"] >= 13).astype(int)
    # Is married flag
    df["is_married"] = df["marital_status"].isin(
        ["Married-civ-spouse", "Married-AF-spouse"]
    ).astype(int)
    # Has capital activity
    df["has_capital"] = ((df["capital_gain"] > 0) | (df["capital_loss"] > 0)).astype(int)
    return df


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self, num_cols, cat_cols):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ], remainder="drop")
        return preprocessor

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Data Transformation started.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = clean_data(train_df)
            test_df = clean_data(test_df)

            train_df = engineer_features(train_df)
            test_df = engineer_features(test_df)

            # Extended feature lists after engineering
            extra_num = ["net_capital", "work_intensity", "high_education", "is_married", "has_capital"]
            all_num = NUMERICAL_COLS + extra_num

            # Only keep cols that exist
            available_num = [c for c in all_num if c in train_df.columns]
            available_cat = [c for c in CATEGORICAL_COLS if c in train_df.columns]

            X_train = train_df.drop(columns=[TARGET_COL])
            y_train = train_df[TARGET_COL]
            X_test = test_df.drop(columns=[TARGET_COL])
            y_test = test_df[TARGET_COL]

            preprocessor = self.get_preprocessor(available_num, available_cat)
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_path, preprocessor)
            logger.info(f"Preprocessor saved: {self.config.preprocessor_path}")

            return (
                X_train_arr, y_train.values,
                X_test_arr, y_test.values,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)