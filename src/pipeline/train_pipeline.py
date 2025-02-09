import sys
import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    """
    Orchestrates the entire ML pipeline:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """

    def __init__(self) -> None:
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        """
        Executes the ML pipeline and logs progress.
        """
        try:
            logging.info("Starting data ingestion process")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            logging.info("Starting data transformation process")
            train_arr, test_arr, preprocessor_file_path = self.data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            logging.info("Starting model training process")
            r2_score = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr,
                preprocessor_path=preprocessor_file_path
            )

            logging.info(f"Training completed successfully. Model R² Score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)
