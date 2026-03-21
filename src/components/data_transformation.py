import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logger  
from src.exception import CustomException  
from dataclasses import dataclass
from src.utils import save_object  

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logger.info("Data transformation started")
            
            # ✅ YOUR EXACT COLUMNS (from debug)
            num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
            cat_features = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_features),
                ('cat_pipeline', cat_pipeline, cat_features)
            ])
            
            logger.info(f"✅ Preprocessor: {len(num_features)} num + {len(cat_features)} cat = 11 total")
            return preprocessor
            
        except Exception as e:
            logger.error("Error creating preprocessor")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Data transformation initiated")
            
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
            
            # Skip outliers
            logger.info("✅ Outlier treatment skipped - proceeding to preprocessing")
            
            # Create preprocessor
            preprocess_obj = self.get_data_transformer_object()
            
            # Split features/target
            target_column = "income"
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Transform
            X_train_transformed = preprocess_obj.fit_transform(X_train)
            X_test_transformed = preprocess_obj.transform(X_test)
            
            # Combine arrays
            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]
            
            # Save preprocessor
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocess_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocess_obj
            )
            
            logger.info("✅ Data transformation completed")
            logger.info(f"✅ Train array shape: {train_arr.shape}")
            logger.info(f"✅ Test array shape: {test_arr.shape}")
            return (train_arr, test_arr, self.data_transformation_config.preprocess_obj_file_path)
            
        except Exception as e:
            logger.error("Data transformation failed")
            raise CustomException(e, sys)
