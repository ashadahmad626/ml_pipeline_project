import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_source=None):
        """
        Load data from a CSV file or use the built-in Adult Census Income dataset.
        """
        logger.info("Data Ingestion started.")
        try:
            if data_source and os.path.exists(data_source):
                df = pd.read_csv(data_source)
                logger.info(f"Loaded data from: {data_source}")
            else:
                # Download Adult Census Income dataset
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
                column_names = [
                    "age", "workclass", "fnlwgt", "education", "education_num",
                    "marital_status", "occupation", "relationship", "race",
                    "sex", "capital_gain", "capital_loss", "hours_per_week",
                    "native_country", "income"
                ]
                df = pd.read_csv(url, names=column_names, sep=",\\s*", engine="python", na_values="?")
                logger.info("Loaded Adult Census Income dataset from UCI repository.")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved: {self.config.raw_data_path}")

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["income"])
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)