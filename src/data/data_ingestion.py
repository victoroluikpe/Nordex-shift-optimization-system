import pandas as pd
import numpy as np
from src.logger import configure_logger
from src.exception import MyException
from config.constant import database_path, data_artifact

import sqlite3
import os

import logging

logging = configure_logger()


def load_data():
    """
    loading the data from the database file , validate the schema, and retun the database
    Returns:
        pd.DataFrame: this function return a pandas dataframe containaing the data loaded from the database file.
        the dataframe is validated against the expected schema to ensure that it has correct column and data types

    """
    try:
        logging.info("starting data loading process from the database file")
        # connect to the sqlite database
        conn = sqlite3.connect(database_path)
        shift_data = pd.read_sql("SELECT * FROM shiftPerformance", conn)
        conn.close()

        os.makedirs(data_artifact, exist_ok=True)
        shift_data.to_csv(os.path.join(data_artifact, "ingested_data.csv"), index= False)

        logging.info(f"First rows of the ingested data:\n{shift_data.head()}")
        logging.info("Data loading process complted successfull")



        return shift_data

    except Exception as e:
        logging.error(f"Error occured during data ingestion {e}")
        raise MyException(e)


#load_data()