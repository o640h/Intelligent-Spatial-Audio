import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "rms_mean",
    "centroid_mean",
    "zcr_mean",
    "rolloff_mean",
    "low_freq_ratio",
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def stable_rng(track_name: str, stem_name: str):
    seed_text = f"{track_name}_{stem_name}"
    seed = int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def base_rule_decision(row):
    low_freq_ratio = row["low_freq_ratio"]
    centroid = row["centroid_mean"]
    zcr = row["zcr_mean"]
    file_name = row["file"].lower()

    if "bass" in file_name or low_freq_ratio > 0.60:
        pan = 0.00
        width = 0.10
        depth = 0.35

    elif "vocals" in file_name:
        pan = 0.00
        width = 0.18
        depth = 0.22

    elif "drums" in file_name:
        pan = 0.00
        width = 0.55 if centroid > 6000 else 0.45
        depth = 0.28

    else:
        pan = 0.00
        width = 0.70 if centroid > 3000 else 0.55
        depth = 0.45 if zcr < 0.08 else 0.35

    return pan, width, depth


def apply_subtle_variation(row, pan, width, depth, track_name):
    stem_name = row["file"]
    rng = stable_rng(track_name, stem_name)
    file_name = stem_name.lower()

    if "bass" in file_name:
        pan_jitter = rng.uniform(-0.01, 0.01)
        width_jitter = rng.uniform(-0.01, 0.01)
        depth_jitter = rng.uniform(-0.02, 0.02)

    elif "vocals" in file_name:
        pan_jitter = rng.uniform(-0.03, 0.03)
        width_jitter = rng.uniform(-0.03, 0.03)
        depth_jitter = rng.uniform(-0.03, 0.03)

    elif "drums" in file_name:
        pan_jitter = rng.uniform(-0.04, 0.04)
        width_jitter = rng.uniform(-0.05, 0.05)
        depth_jitter = rng.uniform(-0.03, 0.03)

    else:
        pan_jitter = rng.uniform(-0.06, 0.06)
        width_jitter = rng.uniform(-0.05, 0.05)
        depth_jitter = rng.uniform(-0.04, 0.04)

    pan = clamp(pan + pan_jitter, -1.0, 1.0)
    width = clamp(width + width_jitter, 0.0, 1.0)
    depth = clamp(depth + depth_jitter, 0.0, 1.0)

    return pan, width, depth


def ml_refine(row, model):
    X = pd.DataFrame([{
        "rms_mean": row["rms_mean"],
        "centroid_mean": row["centroid_mean"],
        "zcr_mean": row["zcr_mean"],
        "rolloff_mean": row["rolloff_mean"],
        "low_freq_ratio": row["low_freq_ratio"],
    }])

    pred = model.predict(X[FEATURE_COLUMNS])[0]
    return float(pred[0]), float(pred[1]), float(pred[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features CSV")
    parser.add_argument("--outfile", default="outputs/spatial/spatial_params.csv", help="Output CSV")
    parser.add_argument("--track", default="default_track", help="Track name for deterministic variation")
    parser.add_argument("--model", default=None, help="Optional trained ML model path")
    parser.add_argument("--ml_blend", type=float, default=0.2, help="Blend factor for ML refinement")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    rows = []

    ml_model = None
    if args.model:
        ml_model = joblib.load(args.model)

    for _, row in df.iterrows():
        pan, width, depth = base_rule_decision(row)
        pan, width, depth = apply_subtle_variation(row, pan, width, depth, args.track)

        ml_pan = ml_width = ml_depth = None

        if ml_model is not None:
            pred_pan, pred_width, pred_depth = ml_refine(row, ml_model)

            pan = (1 - args.ml_blend) * pan + args.ml_blend * pred_pan
            width = (1 - args.ml_blend) * width + args.ml_blend * pred_width
            depth = (1 - args.ml_blend) * depth + args.ml_blend * pred_depth

            ml_pan, ml_width, ml_depth = pred_pan, pred_width, pred_depth

        pan = clamp(pan, -1.0, 1.0)
        width = clamp(width, 0.0, 1.0)
        depth = clamp(depth, 0.0, 1.0)

        rows.append({
            "file": row["file"],
            "pan": pan,
            "width": width,
            "depth": depth,
            "ml_pan_pred": ml_pan,
            "ml_width_pred": ml_width,
            "ml_depth_pred": ml_depth,
        })

    out_df = pd.DataFrame(rows)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print(out_df)
    print(f"\nSaved spatial parameters to: {outfile}")


if __name__ == "__main__":
    main()