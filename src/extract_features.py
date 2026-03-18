import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def compute_features(audio_path: Path) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Zero-crossing rate (rough noisiness / percussiveness proxy)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Low-frequency energy ratio
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_bin_mask = freqs < 200
    low_energy = np.sum(stft[low_bin_mask, :])
    total_energy = np.sum(stft) + 1e-10
    low_freq_ratio = float(low_energy / total_energy)

    return {
        "file": audio_path.name,
        "sample_rate": sr,
        "duration_sec": float(len(y) / sr),
        "rms_mean": rms_mean,
        "centroid_mean": centroid_mean,
        "zcr_mean": zcr_mean,
        "rolloff_mean": rolloff_mean,
        "low_freq_ratio": low_freq_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stemdir", required=True, help="Directory containing separated stem WAVs")
    parser.add_argument("--outfile", default="outputs/features/features.csv", help="CSV output path")
    args = parser.parse_args()

    stemdir = Path(args.stemdir)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for wav_file in sorted(stemdir.glob("*.wav")):
        features = compute_features(wav_file)
        rows.append(features)
        print(f"Processed {wav_file.name}")

    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)

    print("\nSaved features to:", outfile)
    print(df)


if __name__ == "__main__":
    main()