import os
import pickle
import sys
import numpy as np
import pandas as pd
from src.logger import logger
from src.exception import CustomException

def save_object(file_path: str, obj) -> None:
    """Save object as pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Model saved at: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise CustomException(e, sys)
