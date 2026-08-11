from src.logger import configure_logger
import yaml
import sys
import os
from src.exception import MyException

def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise MyException(e, sys)