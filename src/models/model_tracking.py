import dagshub
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger import configure_logger
from src.exception import MyException
from src.utils.mlflow_setup import setup_mlfow_connection
import sys
import os
import mlflow
from src.utils.model_utils import get_existing_model_metrics
logging = configure_logger()

class ModelTracker:
    def __init__(self):
        self.client = MlflowClient()
        self.registered_model_name = "NordexShiftOptimizationModel"


    def push_model(self, model, r2_score: float, mae_score: float) -> bool:
        """
        pushing model to mlflow registry and promote to production.
        """
        try:
            logging.info("Checking existing model performance in MLFLOW")
            existing_model_mae, existing_model_r2 = get_existing_model_metrics(
                self.registered_model_name
            )


            push_new_model = False

            # Model comparison
            if existing_model_r2 is None:
                push_new_model = True
                logging.info("No existing model found, will register a new model...")
            elif r2_score > existing_model_r2:
                push_new_model = True
                logging.info(f"the new model performed better than the existing model..")
            elif r2_score == existing_model_r2 and mae_score < existing_model_mae:
                push_new_model = True
                logging.info("Both model has equal r2_score but the new model performed better interm of mae score")
            else:
                logging.info("Existing model has better performance that the new model, skipping push..")
                return False

            with mlflow.start_run():
                logging.info("logging model to mlflow...")
                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("mae_score", mae_score)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.registered_model_name
                )

                logging.info("model logged successfully.")

                return True   
        except Exception as e:
            logging.error(f"error occured during model pushing...{e}")
            raise MyException(e, sys)