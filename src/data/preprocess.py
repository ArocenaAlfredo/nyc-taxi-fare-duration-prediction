from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """
    Returns the project root directory.
    This file is located in src/data/, so we go two levels up.
    """
    return Path(__file__).resolve().parents[2]


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Loads the raw NYC Yellow Taxi parquet file.
    """
    df = pd.read_parquet(input_path)
    return df
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds target and time-based features.
    """
    df = df.copy()

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day"] = df["tpep_pickup_datetime"].dt.day
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)

    return df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing invalid records and extreme outliers.
    Keeps only May 2022 pickup records.
    """
    df = df.copy()

    valid_ratecodes = [1, 2, 3, 4, 5, 6]

    df_clean = df[
        (df["fare_amount"] > 0) &
        (df["fare_amount"] <= 200) &
        (df["trip_duration_minutes"] >= 1) &
        (df["trip_duration_minutes"] <= 180) &
        (df["trip_distance"] > 0) &
        (df["trip_distance"] <= 100) &
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6) &
        (df["RatecodeID"].isin(valid_ratecodes)) &
        (df["tpep_pickup_datetime"] >= "2022-05-01") &
        (df["tpep_pickup_datetime"] < "2022-06-01")
    ].copy()

    return df_clean
def main() -> None:
    """
    Runs the full preprocessing pipeline.
    """
    project_root = get_project_root()

    input_path = project_root / "data" / "raw" / "yellow_tripdata_2022-05.parquet"
    output_path = project_root / "data" / "processed" / "yellow_tripdata_2022-05_clean.parquet"

    print("Loading raw data...")
    df = load_raw_data(input_path)
    print(f"Raw data shape: {df.shape}")

    print("Adding features...")
    df = add_features(df)

    print("Cleaning data...")
    df_clean = clean_data(df)
    print(f"Clean data shape: {df_clean.shape}")

    print("Saving processed data...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()