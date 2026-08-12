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
from src.models.model_training import ModelTrainer
logger = configure_logger()


def start_model_training():
    try:
        shift_data = load_data()
        Validated_data = validate_data(shift_data)
        Processed_data = start_data_preprocessing(Validated_data)
        X_train, X_test, y_train, y_test = start_feature_engineering(Processed_data)
        print("feature engineering completed...")
        logging.info("Intializing model training...")
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        Pipeline = trainer.train_model()
        r2, mae = trainer.evaluate_model()
        logging.info("model Training completed and preparing to be tracked..")
        model_tracker = ModelTracker()
        was_registered = model_tracker.push_model(
            model = Pipeline,
            r2_score= r2,
            mae_score= mae
        )
        if was_registered:
            logging.info("New pipeline registered to mlflow..")
        else:
            logging.info("Existing model has a better performance than the new pipeline.. ")

        return r2, mae, Pipeline        
                     
    except Exception as e:
        raise MyException(e, sys)



start_model_training()