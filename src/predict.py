import cloudpickle as cp
import os
import sys
import pandas as pd
from typing import Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.preprocess.SC_preprocess import transform_preprocessor
from src.preprocess.SD_preprocess import handler
from src.preprocess.YR_preprocess import transform_preprocessing

custom_globals = {
    #'preprocess_data': preprocess_data,
    #'Preprocessor': Preprocessor,
    #'preprocess_dtypes': preprocess_dtypes,
    'handler': handler,
    'transform_preprocessor': transform_preprocessor,
    'transform_preprocessing': transform_preprocessing
}

def predict_market_demand(data: dict[str, Any]):

    new_data = {
        "Year": [data["year"]],  
        "Month": [data["month"]],  
        "Crop": [data["crop"]],
        "Region": [data["region"]]
    }

    new_data  = pd.DataFrame(new_data)  # Now it behaves like the second input

    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'preprocessor_SD.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'model_SD.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
    new_columns = model.feature_names_in_
    # Transform and make predictions
    data_transformed = pd.DataFrame(preprocessor.transform(new_data), columns=new_columns)
    predicted_demand = model.predict(data_transformed)
    return float(predicted_demand[0])

def predict_compatibility(data: dict[str, Any]):

    classifier_input = {
    "Crop_Type": [data["crop"]],  # Ensure single crop
    "Soil_Type": [data["soil_type"]],
    "Farm_Size_Acres": [data["farm_size_acres"]],
    "Irrigation_Available": [data["irrigation_available"]],
    "Soil_pH": [data["soil_pH"]],
    "Soil_Nitrogen": [data["soil_nitrogen"]],
    "Soil_Organic_Matter": [data["soil_organic_matter"]],
    "Temperature": [data["temperature"]],
    "Rainfall": [data["rainfall"]],
    "Humidity": [data["humidity"]],
    }
    classifier_input = pd.DataFrame(classifier_input)
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'preprocessor_SC.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'model_SC.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
    # Transform and make prediction
    data_transformed = preprocessor.transform(classifier_input)
    proba = model.predict_proba(data_transformed)[0][1] 
    return proba

def predict_yield(data: dict[str, Any]):

    yield_input = {
        "Year": [data["year"]],
        "Month": [data["month"]],
        "Crop": [data["crop"]],  # Ensure single crop
        "Region": [data["region"]],
        "Temperature": [data["temperature"]],
        "Rainfall": [data["rainfall"]],
        "Humidity": [data["humidity"]],
        "Soil_pH": [data["soil_pH"]],
        "Soil_Nitrogen": [data["soil_nitrogen"]],
        "Soil_Phosphorus": [data["soil_phosphorus"]],
        "Soil_Potassium": [data["soil_potassium"]],
        "Fertilizer_Use": [data["fertilizer_use"]],
        "Pesticide_Use": [data["pesticide_use"]],
        "Previous_Year_Yield": [data["previous_year_yield"]],
        "Sowing_To_Harvest_Days": [data["sowing_to_harvest_days"]],
    }
    yield_input = pd.DataFrame(yield_input)
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'preprocessor_YR.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'model_YR.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
    # Transform and make predictions
    data_transformed = preprocessor.transform(yield_input)
    predicted_yield = model.predict(data_transformed)
    return predicted_yield
