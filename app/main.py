from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import os
import sys
import asyncio

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.predict import predict_market_demand, predict_compatibility, predict_yield

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Define input schema
class CropRequest(BaseModel):
    year: int
    month: int
    region: str
    temperature: float
    rainfall: float
    humidity: float
    soil_pH: float
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    soil_organic_matter: float
    fertilizer_use: float
    pesticide_use: float
    previous_year_yield: float
    sowing_to_harvest_days: int
    farm_size_acres: float
    irrigation_available: int
    supply_tons: List[float]
    soil_type: str  
    crops: List[str]  

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/recommend_crops")
async def recommend_crops(
    request: CropRequest, 
    weight_compatibility: float = Query(0.5, ge=0.0, le=1.0),
    weight_yield: float = Query(0.5, ge=0.0, le=1.0)
):
    if not request.crops:
        return {"message": "No crops provided for evaluation."}

    crops = request.crops
    supply = request.supply_tons
    crop_scores = []
    
    # Ensure input validation
    if len(supply) != len(crops):
        return {"error": "Mismatch: supply_tons list length must match crops list length."}

    crop_data = request.model_dump()
    crop_data["irrigation_available"] = int(crop_data["irrigation_available"])

    loop = asyncio.get_event_loop()
    
    # Predict demand asynchronously
    demand_tasks = [
    loop.run_in_executor(None, predict_market_demand, {**crop_data, "crop": crop, "supply_tons": supply[i]}) 
    for i, crop in enumerate(crops)
]
    predicted_demand = await asyncio.gather(*demand_tasks)

    # Identify demanded crops
    demanded_crops = [crops[i] for i in range(len(crops)) if predicted_demand[i] > supply[i]]
    
    if not demanded_crops:
        return {"message": "No significant demand detected for any crop."}

    # Predict compatibility and yield asynchronously
    compatibility_tasks = [loop.run_in_executor(None, predict_compatibility, {**crop_data, "crop": crop}) for crop in demanded_crops]
    yield_tasks = [loop.run_in_executor(None, predict_yield, {**crop_data, "crop": crop}) for crop in demanded_crops]

    compatibility_scores, yield_predictions = await asyncio.gather(
        asyncio.gather(*compatibility_tasks),
        asyncio.gather(*yield_tasks)
    )

    # Convert results to float
    compatibility_scores = [float(score) for score in compatibility_scores]
    yield_predictions = [float(y) for y in yield_predictions]

    # Min-Max Scaling for Yield
    min_yield, max_yield = min(yield_predictions), max(yield_predictions)
    if min_yield == max_yield:
        scaled_yields = [1.0] * len(yield_predictions)
    else:
        scaled_yields = [(y - min_yield) / (max_yield - min_yield) for y in yield_predictions]

    # Compute final scores
    for i, crop in enumerate(demanded_crops):
        final_score = (weight_compatibility * compatibility_scores[i]) + (weight_yield * scaled_yields[i])
        crop_scores.append({"crop": crop, "score": final_score})

    # Rank crops
    ranked_crops = sorted(crop_scores, key=lambda x: x["score"], reverse=True)

    return ranked_crops       


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
