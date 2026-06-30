from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="NYC Taxi Fare and Duration Prediction API",
    description="API for predicting NYC taxi trip fare and duration.",
    version="1.0.0",
)


def get_project_root() -> Path:
    """
    Returns the project root directory.
    This file is located in src/api/, so we go two levels up.
    """
    return Path(__file__).resolve().parents[2]
class TaxiTripInput(BaseModel):
    """
    Input data required to predict taxi fare and trip duration.
    """

    VendorID: int
    passenger_count: float
    trip_distance: float
    RatecodeID: float
    PULocationID: int
    DOLocationID: int
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_day: int = Field(..., ge=1, le=31)
    pickup_weekday: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)

class TaxiTripPrediction(BaseModel):
    """
    Prediction response returned by the API.
    """

    predicted_fare: float
    predicted_duration_minutes: float

def load_models_and_features():
    """
    Loads trained models and feature column order.
    """
    project_root = get_project_root()
    models_dir = project_root / "models"

    fare_model = joblib.load(models_dir / "fare_decision_tree_model.pkl")
    duration_model = joblib.load(models_dir / "duration_decision_tree_model.pkl")

    with open(models_dir / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return fare_model, duration_model, feature_columns


fare_model, duration_model, feature_columns = load_models_and_features()

@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {
        "message": "NYC Taxi Fare and Duration Prediction API is running",
        "status": "ok",
    }

@app.post("/predict", response_model=TaxiTripPrediction)
def predict_trip(trip: TaxiTripInput):
    """
    Predicts fare amount and trip duration for a taxi ride.
    """
    input_data = trip.model_dump()

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    predicted_fare = fare_model.predict(input_df)[0]
    predicted_duration = duration_model.predict(input_df)[0]

    return TaxiTripPrediction(
        predicted_fare=round(float(predicted_fare), 2),
        predicted_duration_minutes=round(float(predicted_duration), 2),
    )