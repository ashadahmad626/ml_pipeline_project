import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils.utils import load_object


WORKCLASS_MAP = {
    "State-gov": 6, "Self-emp-not-inc": 5, "Private": 3,
    "Federal-gov": 0, "Local-gov": 1, "Self-emp-inc": 4,
    "Without-pay": 7, "Never-worked": 2,
}
EDUCATION_MAP = {
    "Bachelors": 13, "HS-grad": 9, "11th": 7, "Masters": 14,
    "9th": 5, "Some-college": 10, "Assoc-acdm": 12, "Assoc-voc": 11,
    "7th-8th": 4, "Doctorate": 16, "Prof-school": 15, "5th-6th": 3,
    "10th": 6, "1st-4th": 2, "Preschool": 1, "12th": 8,
}
MARITAL_MAP = {
    "Never-married": 4, "Married-civ-spouse": 2, "Divorced": 0,
    "Married-spouse-absent": 3, "Separated": 5,
    "Married-AF-spouse": 1, "Widowed": 6,
}
OCCUPATION_MAP = {
    "Adm-clerical": 0, "Exec-managerial": 3, "Handlers-cleaners": 5,
    "Prof-specialty": 9, "Other-service": 7, "Sales": 11,
    "Craft-repair": 2, "Transport-moving": 13, "Farming-fishing": 4,
    "Machine-op-inspct": 6, "Tech-support": 12, "Protective-serv": 10,
    "Armed-Forces": 1, "Priv-house-serv": 8,
}
RELATIONSHIP_MAP = {
    "Not-in-family": 1, "Husband": 0, "Wife": 5,
    "Own-child": 3, "Unmarried": 4, "Other-relative": 2,
}
RACE_MAP = {
    "White": 4, "Black": 2, "Asian-Pac-Islander": 1,
    "Amer-Indian-Eskimo": 0, "Other": 3,
}
SEX_MAP = {"Female": 0, "Male": 1}
NATIVE_COUNTRY_MAP = {
    "United-States": 38, "Cuba": 4, "Jamaica": 22, "India": 18,
    "Mexico": 25, "South": 34, "Puerto-Rico": 32, "Honduras": 15,
    "England": 8, "Canada": 1, "Germany": 10, "Iran": 19,
    "Philippines": 29, "Italy": 21, "Poland": 30, "Columbia": 3,
    "Cambodia": 0, "Thailand": 36, "Ecuador": 6, "Laos": 24,
    "Taiwan": 35, "Haiti": 13, "Portugal": 31,
    "Dominican-Republic": 5, "El-Salvador": 7, "France": 9,
    "Guatemala": 12, "China": 2, "Japan": 23, "Yugoslavia": 40,
    "Peru": 28, "Outlying-US(Guam-USVI-etc)": 27, "Scotland": 33,
    "Trinadad&Tobago": 37, "Greece": 11, "Nicaragua": 26,
    "Vietnam": 39, "Hong": 16, "Ireland": 20,
    "Hungary": 17, "Holand-Netherlands": 14,
}


@dataclass
class CustomDataClass:
    age: int
    workclass: str
    fnlwgt: float
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Convert input to DataFrame compatible with the training pipeline."""
        data = {
            "age": [self.age],
            "workclass": [self.workclass],
            "fnlwgt": [self.fnlwgt],
            "education": [self.education],
            "education_num": [self.education_num],
            "marital_status": [self.marital_status],
            "occupation": [self.occupation],
            "relationship": [self.relationship],
            "race": [self.race],
            "sex": [self.sex],
            "capital_gain": [self.capital_gain],
            "capital_loss": [self.capital_loss],
            "hours_per_week": [self.hours_per_week],
            "native_country": [self.native_country],
        }
        return pd.DataFrame(data)


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, df: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Feature engineering (must match training)
            df = df.copy()
            df["net_capital"] = df["capital_gain"] - df["capital_loss"]
            df["work_intensity"] = df["hours_per_week"] * df["age"]
            df["high_education"] = (df["education_num"] >= 13).astype(int)
            df["is_married"] = df["marital_status"].isin(
                ["Married-civ-spouse", "Married-AF-spouse"]
            ).astype(int)
            df["has_capital"] = ((df["capital_gain"] > 0) | (df["capital_loss"] > 0)).astype(int)

            arr = preprocessor.transform(df)
            pred = model.predict(arr)
            prob = model.predict_proba(arr)[:, 1] if hasattr(model, "predict_proba") else None
            return pred, prob

        except Exception as e:
            raise CustomException(e, sys)