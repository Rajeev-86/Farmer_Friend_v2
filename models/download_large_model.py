import gdown
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

#pickle
#MODEL_URL = "https://drive.google.com/uc?id=1rnoQhjp59h89j6vK2YjcqBYXTNwMc0jf"

#dill
#MODEL_URL = "https://drive.google.com/uc?id=1-GLGhjhgk7B582InVAAX4sIOWMNFK3er"

#cloudpickle
#MODEL_URL = "https://drive.google.com/uc?id=1-EKQrfbvBreVZuiNT3rIWuibuF2Urezz"

#final_model
MODEL_URL = "https://drive.google.com/uc?id=1A8RgnFn0GAiQgbK6ioT-OjXWcw23ppBU"

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'final_model_SD.pkl')

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading large model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
