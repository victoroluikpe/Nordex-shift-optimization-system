from src.logger import configure_logger
from src.exception import MyException
import pandas as pd
import os
import sys

from src.data.data_ingestion import load_data
logger = configure_logger()
class Datavalidation:
    """
    This class is responsible for validating the data loaded from the database file'
    it checks if the data has the expected schema including the correct column and data types
        
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("Data validation class initialized sucessfully.")

    def check_empty_data(self):
        """ checks if dataframe is empty and raise an exception if it is"""
        logger.info("checking if the dataframe is empty")

        if self.df is None or self.df.empty:
            logger.error("The dataframe is empty, No data to validate.")
            raise MyException("The dataframe is empty. No data to validate")
        logger.info("The data frame is not empty. processing with validation")

    def checking_for_missing_values(self):
        logger.info("checking for missing values in the datafram")
        missing_value = self.df.isna().sum()

        if missing_value.sum() > 0:
              logger.warning(f"Missing value found in the dataframe: \n {missing_value}")
        else:
             logger.info("No missing value found in the dataframe.")
        return missing_value

    def checking_for_duplicates(self):
        logger.info("checking for duplicate rows in the dataframe")
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:
              logger.warning(f"Duplicates found in the dataframe: {duplicates} duplicate rows.")
        else:
             logger.info("No duplicate rows found in the dataframe.")
        return duplicates

def validate_data(df: pd.DataFrame):
    """
    This function validate the data loaded from the database file.
    it checks if data has expected schema including the correct columns and data types.

    Args:
        df (pd.DataFrame): The dataframe to be validated.
    Returns:
        pd.dataFrame: The validated dataframe
     
    """
    try:
       logger.info("starting data validation process")
       validator = Datavalidation(df)
       validator.check_empty_data()
       validator.checking_for_missing_values()
       validator.checking_for_duplicates()

       logger.info("Data validation process completed successfully")

       return df
    except Exception as e:
     logger.error(f"Error occured during data validation: {e}")
     raise MyException(e, sys)

    

#shift_data = load_data()
#validate_data(shift_data)
           