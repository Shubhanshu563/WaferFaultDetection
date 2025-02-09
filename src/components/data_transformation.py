import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Defines the path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This class is responsible for:
    1. Data preprocessing (handling missing values, scaling).
    2. Handling class imbalance using SMOTETomek.
    3. Saving the preprocessor object for future use.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline with:
        - NA value replacement with np.nan.
        - Missing value imputation (replacing NaN with 0).
        - Scaling using RobustScaler.
        
        Returns:
            preprocessor (Pipeline): Preprocessing pipeline.
        """
        try:
            logging.info("Creating data transformation pipeline")

            # Define custom function to replace 'na' with np.nan
            replace_na_with_nan = lambda X: np.where(X == 'na', np.nan, X)

            # Define steps for the preprocessor pipeline
            preprocessor = Pipeline(
                steps=[
                    ('nan_replacement', FunctionTransformer(replace_na_with_nan)),
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', RobustScaler())
                ]
            )

            logging.info("Data transformation pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Orchestrates the data transformation process:
        - Loads train and test data.
        - Applies preprocessing.
        - Handles class imbalance using SMOTETomek.
        - Saves the preprocessor object.
        
        Args:
            train_path (str): Path to training dataset.
            test_path (str): Path to testing dataset.

        Returns:
            Tuple: Transformed training and testing arrays, preprocessor file path.
        """
        try:
            logging.info("Reading train and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.get_data_transformer_object()

            # Define target column and mapping
            target_column_name = "class"
            target_column_mapping = {'+1': 0, '-1': 1}

            # Splitting input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name].map(target_column_mapping)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name].map(target_column_mapping)

            logging.info("Applying preprocessing transformations")
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            logging.info("Handling class imbalance using SMOTETomek")
            smt = SMOTETomek(sampling_strategy="auto") 

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

            logging.info("Combining features and target labels into final dataset")
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            logging.info(f"Saving preprocessor object at {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)

            logging.info("Data transformation completed successfully")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
