from pathlib import Path
import pandas as pd

feature_dir = Path("outputs/features")
out_path = Path("data/training/spatial_training_data.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = []

for csv_file in sorted(feature_dir.glob("*_features.csv")):
    track_id = csv_file.stem.replace("_features", "")
    df = pd.read_csv(csv_file)
    df.insert(0, "track_id", track_id)
    rows.append(df)

combined = pd.concat(rows, ignore_index=True)

# Empty target columns to fill manually
combined["target_pan"] = ""
combined["target_width"] = ""
combined["target_depth"] = ""

combined.to_csv(out_path, index=False)
print(f"Saved combined training CSV to: {out_path}")
print(combined)