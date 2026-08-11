import pandas as pd
import numpy as np
import sys
import os
from src.utils.schema_loader import read_yaml
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.data.data_preprocessing import DataPreprocessing, start_data_preprocessing
from config.constant import SCHEMA_PATH

from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

class Feature_Engineering:
    def __init__(self, df: pd.DataFrame):
        self.shift_data = df
        logging.info("feature engineering initialized")

    def Engineer_Features(self):
        try:
            logging.info("starting feature engineering...")
            # parsing date
            # feature engineering
            self.shift_data['start_time'] = pd.to_datetime(self.shift_data['start_time'])
            self.shift_data['end_time'] = pd. to_datetime(self.shift_data['end_time'])

            # fixing overnight shifts where end_time is less than start_time
            mask = self.shift_data['end_time'] < self.shift_data['start_time']
            self.shift_data.loc[mask, 'end_time'] = self.shift_data.loc[mask, 'end_time'] + pd.Timedelta(days = 1)

            # 1, shift duration in hours
            self.shift_data['shift_duration'] = (self.shift_data['end_time'] - self.shift_data['start_time']).dt.total_seconds() / 3600

            # defect ratio
            self.shift_data['defect_rate'] = self.shift_data['defect_count'] / self.shift_data['units_produced'].replace(0, pd.NA)

            # downtime ratio
            self.shift_data['downtime_ratio'] = self.shift_data['downtime_minutes'] / (self.shift_data['shift_duration'] * 60)

            # temporal feature features
            self.shift_data['date'] = pd.to_datetime(self.shift_data['date']) # first do this conversion for the nxt line to work!
            self.shift_data['day_of_week'] = self.shift_data['date'].dt.dayofweek

            logging.info(self.shift_data.head())

            return self.shift_data
        except Exception as e:
            logging.error(f"error occurred while engineering new features {e}")
            raise MyException(e, sys)
        ### Feature selection
    def Feature_Selection(self):
        try:
            schema = read_yaml(SCHEMA_PATH)
            columns_to_drop = schema['columns']['columns_to_drop']

            existing_cols = [column for column in columns_to_drop if column in self.shift_data.columns]
            self.shift_data.drop(columns = existing_cols, inplace= True)

            processor = DataPreprocessing(self.shift_data)
            self.shift_data = processor.remove_duplicates()

            logging.info("Dropped unwanted columns and removed duplicates")
            logging.info(self.shift_data.head())
            return self.shift_data

        except Exception as e:
            logging.error(f"error occurred during feature selection")
            raise MyException(e, sys)


    def Feature_engineering_Engine(self):
        try:
            self.shift_data = self.Engineer_Features()
            self.shift_data = self.Feature_Selection()
            logging.info("feature engineering completed")
            return self.shift_data

        except Exception as e:
            raise MyException(e, sys)

def start_feature_engineering(shift_data: pd.DataFrame):
    try:
        engineer = Feature_Engineering(shift_data)
        shift_data = engineer.Feature_engineering_Engine()

        ## calling the splitting processor
        processor = DataPreprocessing(shift_data)
        X_train, X_test, y_train, y_test = processor.train_test_splitting(shift_data)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("error occured during features engineering initalization....")
        raise MyException (e, sys)
            
               
            
         
shift_data = load_data()
Validated_data = validate_data(shift_data)
Processed_data = start_data_preprocessing(Validated_data)
X_train, X_test, y_train, y_test = start_feature_engineering(Processed_data)
print("feature engineering completed...")
print (X_train.head())
print(X_test.head)

start_feature_engineering