"""
Train an XGBoost model for next-day stock price prediction.

Expected input data (CSV) in data/stock_prices.csv with columns like:
    date, open, high, low, close, volume

The script:
    - loads and sorts the data
    - engineers simple time-series features (returns, moving averages, volatility)
    - trains an XGBoost regressor to predict next-day close
    - reports MAE on a hold-out set
    - saves the trained model to models/xgb_stock_model.pkl
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib


DATA_PATH_DEFAULT = "data/stock_prices.csv"
MODEL_DIR = "models"
MODEL_FILENAME = "xgb_stock_model.pkl"


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1-day return
    df["return_1d"] = df["close"].pct_change()

    # Moving averages
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()

    # Simple volatility measure
    df["volatility_5"] = df["close"].pct_change().rolling(window=5).std()

    # Target: next-day close
    df["target"] = df["close"].shift(-1)

    # Drop rows with NaNs from rolling/shift operations
    df.dropna(inplace=True)
    return df


def train_model(df: pd.DataFrame):
    feature_cols = ["close", "return_1d", "ma_5", "ma_10", "volatility_5"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[INFO] Validation MAE: {mae:.4f}")

    return model


def save_model(model, output_dir: str = MODEL_DIR, filename: str = MODEL_FILENAME):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    print(f"[INFO] Saved model to {path}")


def main(args):
    df = load_data(args.data_path)
    df = add_features(df)
    model = train_model(df)
    save_model(model, MODEL_DIR, MODEL_FILENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost stock price model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH_DEFAULT,
        help="Path to CSV with historical stock prices.",
    )
    args = parser.parse_args()
    main(args)
