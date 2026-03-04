import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model


def load_wav(path: Path):
    """Load WAV as float32 in shape [channels, samples]."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T  # [channels, samples]
    return audio, sr


def save_wav(path: Path, audio: np.ndarray, sr: int):
    """Save float32 audio in shape [channels, samples] to WAV (PCM16)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sr, subtype="PCM_16")


def ensure_wav_with_ffmpeg(infile: Path, processed_dir: Path, sr: int = 44100) -> Path:
    """
    Convert any audio format supported by ffmpeg into a standard WAV:
    - stereo (-ac 2)
    - sample rate sr (-ar)
    - PCM 16-bit
    Returns path to processed wav.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_wav = processed_dir / f"{infile.stem}_sr{sr}.wav"

    # If already a wav at the target name, reuse it (fast re-runs)
    if out_wav.exists():
        return out_wav

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(infile),
        "-ac",
        "2",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH, then restart VS Code."
        ) from e
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg conversion failed. ffmpeg stderr (tail):\n{err}") from e

    return out_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input audio path (mp3/wav/flac/etc.)")
    parser.add_argument("--outdir", default="outputs/stems", help="Output directory")
    parser.add_argument("--model", default="htdemucs_ft", help="Demucs model name")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate for preprocessing")
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    processed_wav = ensure_wav_with_ffmpeg(infile, Path("data/processed"), sr=args.sr)

    # Output folder is based on ORIGINAL file name (cleaner)
    outdir = Path(args.outdir) / args.model / infile.stem

    audio_np, sr = load_wav(processed_wav)

    # Torch tensor [batch, channels, samples]
    audio = torch.from_numpy(audio_np).unsqueeze(0)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    model = get_model(args.model)
    model.to(device)

    with torch.no_grad():
        # returns [sources, channels, samples] for batch item 0
        sources = apply_model(model, audio.to(device))[0]

    stem_names = model.sources
    for i, name in enumerate(stem_names):
        stem = sources[i].detach().cpu().numpy()
        save_wav(outdir / f"{name}.wav", stem, sr)
        print(f"Saved {name} -> {outdir / f'{name}.wav'}")

    print(f"\nProcessed input WAV: {processed_wav}")
    print(f"Stems output dir:     {outdir}")


if __name__ == "__main__":
    main()