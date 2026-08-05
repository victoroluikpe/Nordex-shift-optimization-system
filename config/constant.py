import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



database_path = os.path.abspath(os.path.join(BASE_DIR, 'ShiftData.db.db'))
data_artifact = os.path.abspath(os.path.join(BASE_DIR, 'data_artifact', 'data')) 

print(database_path)