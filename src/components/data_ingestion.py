import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException 
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.modrl_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):  
        logger.info("Data ingestion initiated")
        try:
            #notbook/data/income_cleandata.csv
            data_path = os.path.join("notebooks", "data", "income_cleandata.csv")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            logger.info(f"Reading data from: {data_path}")
            data = pd.read_csv(data_path)
            logger.info("Data reading completed")
            
            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")
            
            # Train-test split
            train_set, test_set = train_test_split(
                data, test_size=0.30, random_state=42, shuffle=True
            )
            logger.info("Data split into train and test sets")
            
            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logger.error("Error occurred in data ingestion stage")
            raise CustomException(e, sys) 

# if __name__ == "__main__":
#     try:
#         data_ingestion = DataIngestion()
#         train_path, test_path = data_ingestion.initiate_data_ingestion()
        
#         logger.info(f"Ingestion complete - Train: {train_path}, Test: {test_path}")
        
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         raise CustomException(e, sys)


# Add this to END of your data_ingestion.py (replace existing __main__ block)

if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logger.info(f"Ingestion complete - Train: {train_path}, Test: {test_path}")
        
        # Step 2: Data Transformation (NEW!)
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocess_path = data_transformation.initiate_data_transformation(train_path, test_path)
        
        logger.info(f"FULL PIPELINE COMPLETE!")
        logger.info(f"Train array shape: {train_arr.shape}")
        logger.info(f"Test array shape: {test_arr.shape}")
        logger.info(f"Preprocessor saved: {preprocess_path}")
        
        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise CustomException(e, sys)
    
    

#src/components/data_ingestion.py

