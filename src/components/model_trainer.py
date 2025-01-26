import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, load_object, save_object, upload_file


@dataclass
class ModelTrainerConfig:
    # Configuration class to store the path of the trained model file
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class CustomModel:
    # Custom wrapper class for handling preprocessing and model inference
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        # Transform input features using preprocessing and make predictions
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        # Representation of the model object (for debugging purposes)
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        # String representation of the model object
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Trains multiple models, evaluates them, selects the best-performing model,
        and saves it along with the preprocessing object.
        """
        try:
            logging.info(f"Splitting training and testing input and target feature")

            # Split training and testing arrays into input and target features
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],  # Input features for training
                train_array[:, -1],  # Target variable for training
                test_array[:, :-1],  # Input features for testing
                test_array[:, -1],  # Target variable for testing
            )

            # Dictionary of machine learning models to train and evaluate
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            logging.info(f"Extracting model config file path")

            # Evaluate models and get performance scores
            model_report: dict = evaluate_models(X=x_train, y=y_train, models=models)

            # Get the best model score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Identify the name of the best-performing model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Check if the best model's performance is acceptable
            if best_model_score < 0.6:
                raise Exception("No suitable model found with sufficient performance")

            logging.info(f"Best found model on both training and testing dataset")

            # Load the preprocessing object
            preprocessor_obj = load_object(file_path=preprocessor_path)

            # Wrap the preprocessing object and model in a CustomModel
            custom_model = CustomModel(
                preprocessing_object=preprocessor_obj,
                trained_model_object=best_model
            )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            # Save the CustomModel object to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model
            )

            # Make predictions on the test dataset using the best model
            predicted = best_model.predict(x_test)

            # Calculate R2 score for the model's predictions
            r2_square = r2_score(y_test, predicted)

            # Upload the trained model file to an S3 bucket
            upload_file(
                from_filename=self.model_trainer_config.trained_model_file_path,
                to_filename="model.pkl",
                bucket_name=AWS_S3_BUCKET_NAME
            )

            return r2_square

        except Exception as e:
            # Handle and log exceptions during the model training process
            raise CustomException(e, sys)