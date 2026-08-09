import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



database_path = os.path.abspath(os.path.join(BASE_DIR, 'ShiftData.db.db'))
data_artifact = os.path.abspath(os.path.join(BASE_DIR, 'data_artifact', 'data')) 
SCHEMA_PATH = os.path.abspath(os.path.join(BASE_DIR, 'config', 'schema.yml'))
target_column ="shift_efficiency_score"
#print(database_path)