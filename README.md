# NYC Taxi Fare and Trip Duration Prediction API

End-to-end machine learning project for predicting **NYC yellow taxi fare amount** and **trip duration** using trip-level information available at the beginning of a ride.

The project includes data preprocessing, exploratory analysis, model training, evaluation, real-time inference through a FastAPI service, and Docker containerization.

---

## Project Overview

Accurate fare and duration estimates are critical for taxi operators, mobility platforms, and passengers. This project builds a supervised machine learning system that estimates:

* Expected taxi fare amount
* Expected trip duration in minutes

The solution uses historical NYC yellow taxi trip records and exposes the trained models through an HTTP API for real-time predictions.

---

## Business Context

Taxi and ridesharing services rely on reliable upfront estimates to improve customer experience and operational efficiency.

A prediction system like this can help:

* Estimate fare before the trip starts.
* Estimate expected travel time.
* Support driver decision-making.
* Improve dispatching and trip planning.
* Provide transparency to passengers.
* Support future demand and pricing analysis.

---

## Dataset

The project uses NYC Yellow Taxi Trip Record data from the NYC Taxi & Limousine Commission.

Dataset used for this version:

```text
Yellow Taxi Trip Records - May 2022
```

The raw data includes trip-level information such as:

* Pickup and dropoff timestamps
* Passenger count
* Trip distance
* Pickup and dropoff location IDs
* Rate code
* Fare amount
* Payment-related fields

Only features that would be available at the beginning of the ride are used as model inputs.

---

## Machine Learning Task

This is a supervised regression problem with two prediction targets:

```text
fare_amount
trip_duration_minutes
```

The `trip_duration_minutes` target is created from:

```text
tpep_dropoff_datetime - tpep_pickup_datetime
```

Two separate models are trained:

* One model for fare prediction
* One model for duration prediction

---

## Features

The current model uses the following input features:

```text
VendorID
passenger_count
trip_distance
RatecodeID
PULocationID
DOLocationID
pickup_hour
pickup_day
pickup_weekday
is_weekend
```

Time-based features are extracted from the pickup timestamp.

Payment-related fields such as tips, tolls, total amount, and payment type are excluded to avoid data leakage.

---

## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads the raw NYC taxi trip dataset.
2. Creates trip duration in minutes.
3. Extracts time-based pickup features.
4. Filters invalid records and outliers.
5. Keeps only valid May 2022 trips.
6. Saves the cleaned dataset as a Parquet file.

Main cleaning rules:

```text
fare_amount > 0
fare_amount <= 200

trip_duration_minutes >= 1
trip_duration_minutes <= 180

trip_distance > 0
trip_distance <= 100

passenger_count >= 1
passenger_count <= 6

RatecodeID in [1, 2, 3, 4, 5, 6]

pickup datetime between 2022-05-01 and 2022-06-01
```

Dataset size after preprocessing:

```text
Raw data shape:   3,588,295 rows, 19 columns
Clean data shape: 3,297,667 rows, 25 columns
```

---

## Exploratory Data Analysis

The EDA showed several relevant patterns:

* Taxi demand increases after 6 AM and peaks around 6 PM.
* Fare amount is strongly correlated with trip distance.
* Trip duration is also related to distance but is more variable.
* Duration is harder to predict because it is affected by temporal and location-based traffic patterns.
* Some records contain invalid or extreme values, such as negative fares, zero distances, and unrealistic durations.

Correlation between main variables:

```text
fare_amount vs trip_distance:              0.95
fare_amount vs trip_duration_minutes:      0.86
trip_duration_minutes vs trip_distance:    0.82
```

---

## Models

The following models were trained and evaluated:

* Linear Regression
* Decision Tree Regressor

A temporal train/test split was used:

```text
Train: trips before 2022-05-25
Test:  trips from 2022-05-25 onward
```

This evaluation setup simulates a realistic scenario where the model is trained on past trips and evaluated on future trips.

---

## Results

| Target                | Model             |   MAE |  RMSE |    R² |
| --------------------- | ----------------- | ----: | ----: | ----: |
| fare_amount           | Linear Regression | 2.269 | 3.672 | 0.923 |
| fare_amount           | Decision Tree     | 1.460 | 2.594 | 0.961 |
| trip_duration_minutes | Linear Regression | 5.611 | 7.777 | 0.647 |
| trip_duration_minutes | Decision Tree     | 4.004 | 6.243 | 0.773 |

The Decision Tree Regressor achieved the best performance for both targets.

Current best results:

```text
Fare prediction MAE:      1.46 USD
Duration prediction MAE:  4.00 minutes
```

---

## Model Interpretation

Feature importance analysis showed that:

* `trip_distance` is the strongest predictor for both fare and duration.
* `RatecodeID` is especially relevant for fare prediction.
* `pickup_hour` is more important for duration prediction, likely because of traffic patterns.
* Location IDs contribute to duration prediction by capturing pickup and dropoff area effects.

This confirms that fare is mainly driven by distance and fare type, while duration depends more on distance, time, and location patterns.

---

## API

The trained models are exposed through a FastAPI application.

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "NYC Taxi Fare and Duration Prediction API is running",
  "status": "ok"
}
```

### Prediction Endpoint

```http
POST /predict
```

Example request:

```json
{
  "VendorID": 1,
  "passenger_count": 1,
  "trip_distance": 2.5,
  "RatecodeID": 1,
  "PULocationID": 246,
  "DOLocationID": 151,
  "pickup_hour": 14,
  "pickup_day": 25,
  "pickup_weekday": 2,
  "is_weekend": 0
}
```

Example response:

```json
{
  "predicted_fare": 14.43,
  "predicted_duration_minutes": 20.32
}
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
nyc-taxi-fare-duration-prediction/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── api/
│       └── main.py
│
├── reports/
│   └── figures/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
└── .dockerignore
```

---

## Reproducibility

The repository does not include raw datasets or trained model files.

To reproduce the full pipeline, download the NYC Yellow Taxi May 2022 dataset and place it in:

```text
data/raw/yellow_tripdata_2022-05.parquet
```

Then run the preprocessing and training scripts.

---

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run preprocessing

```powershell
python src\data\preprocess.py
```

### 4. Train models

```powershell
python src\models\train.py
```

This will generate:

```text
models/fare_decision_tree_model.pkl
models/duration_decision_tree_model.pkl
models/feature_columns.json
models/model_results.csv
```

### 5. Run a sample prediction

```powershell
python src\models\predict.py
```

---

## Run the API Locally

```powershell
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

---

## Run with Docker

Build the image:

```powershell
docker build -t nyc-taxi-api .
```

Run the container:

```powershell
docker run -p 8000:8000 nyc-taxi-api
```

---

## Run with Docker Compose

The API can also be started using Docker Compose:

```powershell
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

To stop the service:

```powershell
docker compose down
```

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Pydantic
* Uvicorn
* Joblib
* PyArrow
* Jupyter Notebook
* Matplotlib
* Seaborn
* Docker
* Docker Compose

---

## Limitations

This version uses one month of taxi trip data and does not include external data sources.

Current limitations:

* No weather data
* No traffic flow data
* No road closure information
* No event data
* No route-level features
* No geographic distance calculation from coordinates
* No advanced gradient boosting or neural network models yet

---

## Future Improvements

Possible next steps:

* Train with multiple months of data.
* Add weather and traffic features.
* Use taxi zone lookup tables for richer location features.
* Add geospatial features.
* Train Random Forest, XGBoost, LightGBM, or neural network models.
* Add automated tests.
* Add CI/CD workflow.
* Deploy the API to a cloud platform.
* Build a simple frontend for predictions.
* Extend the project to taxi demand forecasting.

---

## Conclusion

This project implements a complete machine learning workflow for NYC taxi fare and trip duration prediction.

The final solution includes:

* Data preprocessing
* Exploratory data analysis
* Supervised regression modeling
* Model evaluation
* Real-time prediction API
* Dockerized deployment setup

The best current models are Decision Tree Regressors, achieving approximately **1.46 USD MAE** for fare prediction and **4.00 minutes MAE** for trip duration prediction.
