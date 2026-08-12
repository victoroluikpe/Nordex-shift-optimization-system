import pandas as pd
import numpy as np
import sys
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

def prediction_pipeline(input_data: dict, model):
    """
    uses the prloaded model from mlflow to make prediction

    Input:
        input_data: diction from the API
        model: preloaded model from MLFLOW(passed from the api)
    Output:
    prediction(list)    
    """
    try:
        logging.info("preparing input data for prediction")
        ## convert the data from dictionary format to Dataframe
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)

        logging.info(f"prediction completed: {prediction}")

        return prediction.tolist()
    except Exception as e:
        logging.error(f"error occured in prediction pipeline {e}")
        raise MyException(e, sys)
