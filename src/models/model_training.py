import mlflow
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.data.data_preprocessing import DataPreprocessing, start_data_preprocessing
from src.features.feature_engineering import start_feature_engineering
import sys
import logging
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import read_yaml
from sklearn.ensemble import GradientBoostingRegressor
from config.constant import SCHEMA_PATH
from src.models.model_tracking import ModelTracker
logger = configure_logger()

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.schema = read_yaml(SCHEMA_PATH)
        self.model_pipeline = None

    def build_training_pipeline(self):
        """ building a full pipeline including  the preprocessing method and model """
        numerical_columns = self.schema["columns"]["numerical_columns"]
        categorical_columns = self.schema["columns"]["categorical_columns"]

        preprocessor = ColumnTransformer(
            transformers= [
            ('num', 'passthrough', numerical_columns),
            ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns)

            ]
        )

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', GradientBoostingRegressor())
            ]
        )

        return pipeline

    def train_model(self):
        try:
            logging.info("started model training pipeline")
            self.model_pipeline = self.build_training_pipeline()
            self.model_pipeline.fit(self.X_train, self.y_train)
            logging.info("pipeline training successfully completed...")
            return self.model_pipeline
        except Exception as e:
            logging.error(f"Error during model training pipeline {e}")
            raise MyException(e, sys)


    def evaluate_model(self):
        try:
            logging.info("Evaluating the trained pipeline....")
            y_pred = self.model_pipeline.predict(self.X_test)
            r2 = sklearn.metrics.r2_score(self.y_test, y_pred)
            mae = sklearn.metrics.mean_absolute_error(self.y_test, y_pred)
            logging.info(f"pipeline evaluation completed with the metrices of R2={r2:.4f}, MAE={mae:.4f}")
            return r2, mae
        except Exception as e:
            logging.error(f"error during model evaluation {str(e)}")
            raise MyException(e, sys)
         
