import sys
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline(data_source=None):
    logger.info("=" * 60)
    logger.info("Training Pipeline Started")
    logger.info("=" * 60)
    try:
        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(data_source)

        # Step 2: Data Transformation
        transformation = DataTransformation()
        X_train, y_train, X_test, y_test, preprocessor_path = \
            transformation.initiate_data_transformation(train_path, test_path)

        # Step 3: Model Training
        trainer = ModelTrainer()
        best_name, best_info, full_report = trainer.initiate_model_training(
            X_train, y_train, X_test, y_test
        )

        logger.info("=" * 60)
        logger.info(f"Pipeline Complete. Best model: {best_name}")
        logger.info(f"  Accuracy : {best_info['accuracy']}")
        logger.info(f"  F1 Score : {best_info['f1_score']}")
        logger.info(f"  ROC-AUC  : {best_info['roc_auc']}")
        logger.info("=" * 60)

        return best_name, best_info, full_report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()