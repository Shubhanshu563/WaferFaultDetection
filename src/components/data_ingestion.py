import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constant import *  # Importing project-specific constants (e.g., database name)
from src.exception import CustomException  # Custom exception handling for better debugging
from src.logger import logging  # Logger for tracking execution flow
from src.utils import export_collection_as_dataframe  # Utility function to fetch data from MongoDB

@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining file paths for data storage.
    This class stores the paths for:
    - Raw data extracted from MongoDB
    - Training dataset
    - Test dataset
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    This class handles the data ingestion process, including:
    - Fetching data from MongoDB and converting it into a DataFrame.
    - Splitting the dataset into training and testing sets.
    - Saving the datasets as CSV files for further processing.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class by setting up file paths using DataIngestionConfig.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Executes the data ingestion pipeline:
        1. Extracts data from MongoDB.
        2. Saves the raw dataset to a CSV file.
        3. Splits the data into training and test sets (80-20 split).
        4. Saves the split datasets as CSV files.
        
        Returns:
            Tuple (train_data_path, test_data_path)
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            # Fetching data from MongoDB as a Pandas DataFrame
            df: pd.DataFrame = export_collection_as_dataframe(
                db_name=MONGO_DATABASE_NAME, collection_name=MONGO_DATABASE_NAME
            )
            logging.info("Successfully exported data collection as a DataFrame")

            # Creating the directory for storing CSV files if it doesn't exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Saving the raw dataset to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Splitting the dataset into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=69)

            # Saving the training set to CSV
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            # Saving the test set to CSV
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info(f"Training data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at: {self.ingestion_config.test_data_path}")

            logging.info("Exiting initiate_data_ingestion method of DataIngestion class")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Raising a custom exception for better debugging in case of an error
            raise CustomException(e, sys)
