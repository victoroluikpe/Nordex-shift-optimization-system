import yaml
import dagshub
from src.exception import MyException
from src.logger import configure_logger
import mlflow.sklearn
from src.utils.mlflow_setup import setup_mlfow_connection
from mlflow.tracking import MlflowClient
logging = configure_logger()
setup_mlfow_connection()

def get_existing_model_metrics(registered_model_name):
    """
    Fetch metrics of the existing registered model in MLFLOW..
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(registered_model_name)
        if not versions:
            return None, None
        existing_model_r2, existing_model_mae = None, None
        for version in versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            r2 = run.data.metrics.get("r2_score")
            mae = run.data.metrics.get("mae_score")

            if r2 is None or mae is None:
                continue
            ## pick the best model based on R2 and mae
            if(existing_model_r2 is None) or (r2 > existing_model_r2) or (r2 == existing_model_r2 and mae < existing_model_mae):
                existing_model_mae = mae
                existing_model_r2 = r2

        return existing_model_mae, existing_model_r2
    except Exception as e:
        logging.warning(f"No existing model found or error occured fetching the metrics {e}")
        return None, None
