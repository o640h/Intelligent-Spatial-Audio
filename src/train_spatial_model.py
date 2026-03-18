import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_COLUMNS = [
    "rms_mean",
    "centroid_mean",
    "zcr_mean",
    "rolloff_mean",
    "low_freq_ratio",
]

TARGET_COLUMNS = [
    "target_pan",
    "target_width",
    "target_depth",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labelled training CSV")
    parser.add_argument("--outmodel", default="models/spatial_refiner.joblib", help="Output model path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    )

    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)

    outmodel = Path(args.outmodel)
    outmodel.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outmodel)

    print("Training complete.")
    print(f"Saved model to: {outmodel}")
    print(f"Training MAE: {mae:.6f}")


if __name__ == "__main__":
    main()