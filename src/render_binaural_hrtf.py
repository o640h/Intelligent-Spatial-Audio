import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfiltfilt, resample_poly


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


def normalize_audio(audio: np.ndarray, peak_target: float = 0.98):
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


def mono_collapse(stereo: np.ndarray):
    mono = np.mean(stereo, axis=1)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def narrow_toward_mono(stereo: np.ndarray, mono_amount: float):
    mono_amount = clamp(float(mono_amount), 0.0, 1.0)
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)

    new_left = (1.0 - mono_amount) * left + mono_amount * mid
    new_right = (1.0 - mono_amount) * right + mono_amount * mid
    return np.stack([new_left, new_right], axis=1).astype(np.float32)


def low_band_mono_amount(file_name: str):
    file_name = file_name.lower()
    if "bass" in file_name:
        return 0.90
    if "drums" in file_name:
        return 0.80
    if "other" in file_name:
        return 0.45
    if "vocals" in file_name:
        return 0.20
    return 0.60


def apply_depth_safe(stereo: np.ndarray, depth: float):
    # Keep this conservative for now
    return stereo.astype(np.float32)


def choose_param_columns(spatial_df: pd.DataFrame):
    cols = set(spatial_df.columns)
    if {"opt_pan", "opt_width", "opt_depth"}.issubset(cols):
        return "opt_pan", "opt_width", "opt_depth"
    if {"pan", "width", "depth"}.issubset(cols):
        return "pan", "width", "depth"
    raise ValueError(
        "Spatial CSV must contain either pan/width/depth or opt_pan/opt_width/opt_depth columns."
    )


def load_hrtf_npz(path: Path):
    data = np.load(path)
    azimuths = data["azimuths"].astype(np.float32)
    hrir_l = data["hrir_l"].astype(np.float32)
    hrir_r = data["hrir_r"].astype(np.float32)
    hrtf_sr = int(data["sample_rate"])
    return azimuths, hrir_l, hrir_r, hrtf_sr


def nearest_hrir(azimuth_deg: float, azimuths: np.ndarray, hrir_l: np.ndarray, hrir_r: np.ndarray):
    idx = int(np.argmin(np.abs(azimuths - azimuth_deg)))
    return hrir_l[idx], hrir_r[idx]


def convolve_binaural_object(signal_mono: np.ndarray, ir_l: np.ndarray, ir_r: np.ndarray):
    out_l = fftconvolve(signal_mono, ir_l, mode="full")[: len(signal_mono)]
    out_r = fftconvolve(signal_mono, ir_r, mode="full")[: len(signal_mono)]
    return np.stack([out_l, out_r], axis=1).astype(np.float32)


def stereo_to_mid_side(stereo: np.ndarray):
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid.astype(np.float32), side.astype(np.float32)


def render_stem_binaural(
    audio: np.ndarray,
    sr: int,
    stem_name: str,
    pan: float,
    width: float,
    depth: float,
    azimuths: np.ndarray,
    hrir_l: np.ndarray,
    hrir_r: np.ndarray,
    hrtf_sr: int,
):
    """
    Advanced minimal HRTF renderer:
    - low band stays stereo-safe / near-centre
    - high band uses HRTF convolution
    - stereo information is preserved through mid/side decomposition
    """
    pan = clamp(float(pan), -1.0, 1.0)
    width = clamp(float(width), 0.0, 1.0)
    depth = clamp(float(depth), 0.0, 1.0)

    # Match HRIR rate to audio if needed
    if hrtf_sr != sr:
        # resample IRs once outside would be faster, but keep simple for now
        pass

    low, high = split_low_high(audio, sr, cutoff_hz=120.0)

    # Low band: keep stable
    low = narrow_toward_mono(low, low_band_mono_amount(stem_name))

    # High band: preserve stereo by decomposing to mid+side
    mid, side = stereo_to_mid_side(high)

    # Map pan [-1,1] to azimuth [-75,75] degrees
    az_mid = float(pan * 75.0)

    # Side can be rendered slightly more extreme
    side_span = 20.0 + 40.0 * width
    az_side_l = clamp(az_mid - side_span, -80.0, 80.0)
    az_side_r = clamp(az_mid + side_span, -80.0, 80.0)

    ir_mid_l, ir_mid_r = nearest_hrir(az_mid, azimuths, hrir_l, hrir_r)
    ir_side_l_l, ir_side_l_r = nearest_hrir(az_side_l, azimuths, hrir_l, hrir_r)
    ir_side_r_l, ir_side_r_r = nearest_hrir(az_side_r, azimuths, hrir_l, hrir_r)

    if hrtf_sr != sr:
        up = sr
        down = hrtf_sr
        ir_mid_l = resample_poly(ir_mid_l, up, down).astype(np.float32)
        ir_mid_r = resample_poly(ir_mid_r, up, down).astype(np.float32)
        ir_side_l_l = resample_poly(ir_side_l_l, up, down).astype(np.float32)
        ir_side_l_r = resample_poly(ir_side_l_r, up, down).astype(np.float32)
        ir_side_r_l = resample_poly(ir_side_r_l, up, down).astype(np.float32)
        ir_side_r_r = resample_poly(ir_side_r_r, up, down).astype(np.float32)

    # Mid object
    mid_out = convolve_binaural_object(mid, ir_mid_l, ir_mid_r)

    # Side as opposing objects
    side_l_obj = convolve_binaural_object(side, ir_side_l_l, ir_side_l_r)
    side_r_obj = convolve_binaural_object(-side, ir_side_r_l, ir_side_r_r)

    # Width controls how much side survives
    high_out = mid_out + width * (side_l_obj - side_r_obj)

    # Blend some original high stereo back in to avoid over-collapse
    orig_high = high.astype(np.float32)
    orig_high = normalize_audio(orig_high, peak_target=10.0)  # no real normalization, just keep dtype
    high_out = 0.65 * high_out + 0.35 * orig_high

    out = low + high_out
    out = apply_depth_safe(out, depth)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stemdir", required=True, help="Directory containing separated stems")
    parser.add_argument("--spatial", required=True, help="CSV with spatial parameters")
    parser.add_argument("--hrtf", required=True, help="Path to HRTF .npz file")
    parser.add_argument("--outdir", default="outputs/binaural_hrtf", help="Output directory")
    parser.add_argument("--mixname", default="binaural_hrtf_mix.wav", help="Final output filename")
    args = parser.parse_args()

    stemdir = Path(args.stemdir)
    outdir = Path(args.outdir)
    processed_stems_dir = outdir / "processed_stems"
    final_mix_path = outdir / args.mixname

    azimuths, hrir_l, hrir_r, hrtf_sr = load_hrtf_npz(Path(args.hrtf))

    spatial_df = pd.read_csv(args.spatial)
    pan_col, width_col, depth_col = choose_param_columns(spatial_df)

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
        processed = render_stem_binaural(
            audio,
            sr,
            stem_file.name,
            params["pan"],
            params["width"],
            params["depth"],
            azimuths,
            hrir_l,
            hrir_r,
            hrtf_sr,
        )

        save_audio(processed_stems_dir / stem_file.name, normalize_audio(processed), sr)
        print(
            f"Processed {stem_file.name} | "
            f"pan={params['pan']:.3f}, width={params['width']:.3f}, depth={params['depth']:.3f}"
        )

        mix += processed

    mix = normalize_audio(mix, peak_target=0.98)
    save_audio(final_mix_path, mix, sample_rate)

    print(f"\nSaved HRTF binaural stems to: {processed_stems_dir}")
    print(f"Saved final HRTF binaural mix to: {final_mix_path}")


if __name__ == "__main__":
    main()