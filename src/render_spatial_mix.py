import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_audio_stereo(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, sr


def save_audio(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def normalize_audio(audio: np.ndarray, peak_target: float = 0.98) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-8
    if peak > peak_target:
        audio = audio * (peak_target / peak)
    return audio.astype(np.float32)


def design_filter(sr: int, cutoff_hz: float, btype: str):
    cutoff_hz = clamp(float(cutoff_hz), 20.0, sr / 2 - 100.0)
    return butter(4, cutoff_hz, btype=btype, fs=sr, output="sos")


def split_low_high(stereo: np.ndarray, sr: int, cutoff_hz: float = 120.0):
    low_sos = design_filter(sr, cutoff_hz, "low")
    high_sos = design_filter(sr, cutoff_hz, "high")

    low_l = sosfiltfilt(low_sos, stereo[:, 0])
    low_r = sosfiltfilt(low_sos, stereo[:, 1])
    high_l = sosfiltfilt(high_sos, stereo[:, 0])
    high_r = sosfiltfilt(high_sos, stereo[:, 1])

    low = np.stack([low_l, low_r], axis=1)
    high = np.stack([high_l, high_r], axis=1)
    return low.astype(np.float32), high.astype(np.float32)


def narrow_toward_mono(stereo: np.ndarray, mono_amount: float):
    """
    mono_amount:
        0.0 = unchanged
        1.0 = fully mono
    """
    mono_amount = clamp(float(mono_amount), 0.0, 1.0)

    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)

    new_left = (1.0 - mono_amount) * left + mono_amount * mid
    new_right = (1.0 - mono_amount) * right + mono_amount * mid
    return np.stack([new_left, new_right], axis=1).astype(np.float32)


def adjust_width_ms(stereo: np.ndarray, width: float):
    """
    Safer M/S width control on already-stereo material.
    width:
        0.0 = narrow/mono
        1.0 = original-ish / slightly enhanced
    """
    width = clamp(float(width), 0.0, 1.0)

    left = stereo[:, 0]
    right = stereo[:, 1]

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    # Conservative mapping:
    # 0.0 -> mono
    # 0.5 -> moderate width
    # 1.0 -> original/slightly enhanced
    side_scale = 0.15 + 1.00 * width
    if width < 0.15:
        side_scale = 0.05 + 0.60 * width

    new_left = mid + side * side_scale
    new_right = mid - side * side_scale
    return np.stack([new_left, new_right], axis=1).astype(np.float32)


def stereo_balance_pan(stereo: np.ndarray, pan: float):
    """
    Preserve stereo image and shift balance slightly.
    pan:
        -1.0 = more left
        +1.0 = more right
    """
    pan = clamp(float(pan), -1.0, 1.0)

    # Conservative balance gains
    left_gain = 1.0 - max(0.0, pan) * 0.35
    right_gain = 1.0 - max(0.0, -pan) * 0.35

    out = stereo.copy().astype(np.float32)
    out[:, 0] *= right_gain if pan > 0 else 1.0
    out[:, 1] *= left_gain if pan < 0 else 1.0

    # Slight boost on intended side, but subtle
    if pan > 0:
        out[:, 1] *= 1.0 + 0.10 * pan
    elif pan < 0:
        out[:, 0] *= 1.0 + 0.10 * abs(pan)

    return out.astype(np.float32)


def apply_depth_safe(stereo: np.ndarray, depth: float):
    """
    Temporary conservative depth stage.
    Only a very small gain trim.
    No reverb, no LPF.
    """
    depth = clamp(float(depth), 0.0, 1.0)
    gain = 1.0 - 0.06 * depth
    return (stereo * gain).astype(np.float32)


def low_band_mono_amount(file_name: str):
    file_name = file_name.lower()

    if "bass" in file_name:
        return 0.95
    if "drums" in file_name:
        return 0.90
    if "other" in file_name:
        return 0.70
    if "vocals" in file_name:
        return 0.40
    return 0.80


def process_stem(audio: np.ndarray, sr: int, stem_name: str, pan: float, width: float, depth: float):
    low, high = split_low_high(audio, sr, cutoff_hz=120.0)

    # Keep low end controlled / mono-safe
    low = narrow_toward_mono(low, low_band_mono_amount(stem_name))

    # Only widen highs/mids
    high = adjust_width_ms(high, width)

    # Apply subtle balance panning after width
    combined = low + high
    combined = stereo_balance_pan(combined, pan)

    # Very conservative depth for now
    combined = apply_depth_safe(combined, depth)

    return combined.astype(np.float32)


def choose_param_columns(spatial_df: pd.DataFrame):
    cols = set(spatial_df.columns)

    if {"opt_pan", "opt_width", "opt_depth"}.issubset(cols):
        return "opt_pan", "opt_width", "opt_depth"

    if {"pan", "width", "depth"}.issubset(cols):
        return "pan", "width", "depth"

    raise ValueError(
        "Spatial CSV must contain either pan/width/depth or opt_pan/opt_width/opt_depth columns."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stemdir", required=True, help="Directory containing separated stems")
    parser.add_argument("--spatial", required=True, help="CSV with spatial parameters")
    parser.add_argument("--outdir", default="outputs/rendered", help="Output directory")
    parser.add_argument("--mixname", default="enhanced_mix.wav", help="Final output filename")
    args = parser.parse_args()

    stemdir = Path(args.stemdir)
    outdir = Path(args.outdir)
    processed_stems_dir = outdir / "processed_stems"
    final_mix_path = outdir / args.mixname

    spatial_df = pd.read_csv(args.spatial)
    pan_col, width_col, depth_col = choose_param_columns(spatial_df)

    if "file" not in spatial_df.columns:
        raise ValueError("Spatial CSV must contain a 'file' column matching stem filenames.")

    spatial_map = {
        str(row["file"]): {
            "pan": float(row[pan_col]),
            "width": float(row[width_col]),
            "depth": float(row[depth_col]),
        }
        for _, row in spatial_df.iterrows()
    }

    stem_files = sorted(stemdir.glob("*.wav"))
    if not stem_files:
        raise FileNotFoundError(f"No WAV stems found in: {stemdir}")

    mix = None
    sample_rate = None

    for stem_file in stem_files:
        if stem_file.name not in spatial_map:
            print(f"Skipping {stem_file.name} (no spatial params found).")
            continue

        audio, sr = load_audio_stereo(stem_file)

        if sample_rate is None:
            sample_rate = sr
            mix = np.zeros_like(audio, dtype=np.float32)
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {stem_file} has {sr}, expected {sample_rate}")

        if len(audio) != len(mix):
            min_len = min(len(audio), len(mix))
            audio = audio[:min_len]
            mix = mix[:min_len]

        params = spatial_map[stem_file.name]
        processed = process_stem(
            audio,
            sr,
            stem_file.name,
            params["pan"],
            params["width"],
            params["depth"],
        )

        save_audio(processed_stems_dir / stem_file.name, normalize_audio(processed), sr)
        print(
            f"Processed {stem_file.name} | "
            f"pan={params['pan']:.3f}, width={params['width']:.3f}, depth={params['depth']:.3f}"
        )

        mix += processed

    mix = normalize_audio(mix, peak_target=0.98)
    save_audio(final_mix_path, mix, sample_rate)

    print(f"\nSaved processed stems to: {processed_stems_dir}")
    print(f"Saved final enhanced mix to: {final_mix_path}")


if __name__ == "__main__":
    main()