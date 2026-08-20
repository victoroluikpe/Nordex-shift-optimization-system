import dagshub
import mlflow
import os

from dotenv import load_dotenv
load_dotenv(override=True)

#def setup_mlfow_connection():
#    dagshub.init(repo_owner='victoroluikpe', 
#            repo_name='Nordex-shift-optimization-system', 
#        mlflow=True)
#    mlflow.set_experiment("Nordex-shift-optimization-production_Model")


def setup_mlflow_connection():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("MLFLOW_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKLING_USERNAME"] = dagshub_token
    os.Environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    Dagshub_Url = "https://dagshub.com"
    Repo_Owner = "victoroluikpe"
    Repo_Name = "Nordex-shift-optimization-system"\
    # set up MLFLOW tracking URI
    mlflow.set_tracking_uri(f"{Dagshub_Url}/{Repo_Owner}/{Repo_Name}.mlflow")
    mlflow.set_experiment("Nordex-shift-optimization-production-Model")



