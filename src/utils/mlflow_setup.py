import dagshub
import mlflow
import os

from dotenv import load_dotenv


def setup_mlfow_connection():
    dagshub.init(repo_owner='victoroluikpe', 
                 repo_name='Nordex-shift-optimization-system', 
                 mlflow=True)
    mlflow.set_experiment("Nordex-shift-optimization-production_Models")