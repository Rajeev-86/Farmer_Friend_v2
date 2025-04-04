import os
import gdown
import sys
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

import os
import gdown

# Define model directories
MODEL_DIRS = {
    "Demand_Predictor": "models/Demand_Predictor",
    "Soil_Climate_Classifier": "models/Soil-Climate_Compatibility_Classifier",
    "Yield_Regression": "models/Yield_Regression",
}

# Define Google Drive file IDs for each model
model_files = {
    "models/Demand_Predictor/model_SD.pkl": "1VJGqyCSNavorYX4bD6kE5xf57NFRy6jZ",
    "models/Demand_Predictor/preprocessor_SD.pkl": "1tR90oCE95OmTuFUUU8SRWLZA2obYFrv5",
    "models/Soil-Climate_Compatibility_Classifier/model_SC.pkl": "1o_Jrxry7jz_OwFf8QhRjS0fr3O-zSSGG",
    "models/Soil-Climate_Compatibility_Classifier/preprocessor_SC.pkl": "1J6iGNXge2lK1JcJB13QKHG8dpMa0ZixZ",
    "models/Yield_Regression/model_YR.pkl": "1BAlU5lWbFLtlNql78TWQTwL23BKFcojs",
    "models/Yield_Regression/preprocessor_YR.pkl": "1sYoQTcShbWslGb9rYvGJruX0l9t7odlg",
}

# Ensure model directories exist
for dir_path in MODEL_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Function to download a file from Google Drive
def download_file(file_id, file_path):
    if not os.path.exists(file_path):  # Download only if not already present
        print(f"Downloading {file_path} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
        print(f"Downloaded {file_path} successfully.")
    else:
        print(f"{file_path} already exists. Skipping download.")

# Download all models and preprocessors
for file_path, file_id in model_files.items():
    download_file(file_id, file_path)

print("✅ All models and preprocessors downloaded successfully!")
