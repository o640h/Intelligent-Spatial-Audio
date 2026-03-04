import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model


def load_wav(path: Path):
    """Load WAV as float32 in shape [channels, samples]."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns shape [samples, channels]
    audio = audio.T  # -> [channels, samples]
    return audio, sr


def save_wav(path: Path, audio: np.ndarray, sr: int):
    """Save float32 audio in shape [channels, samples] to WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects [samples, channels]
    sf.write(str(path), audio.T, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input WAV path")
    parser.add_argument("--outdir", default="outputs/stems", help="Output directory")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir) / args.model / infile.stem

    audio_np, sr = load_wav(infile)

    # Convert to torch tensor: [batch, channels, samples]
    audio = torch.from_numpy(audio_np).unsqueeze(0)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    model = get_model(args.model)
    model.to(device)

    with torch.no_grad():
        sources = apply_model(model, audio.to(device))[0]

    # stems: dict name -> torch tensor [batch, channels, samples]
    stem_names = model.sources

    for i, name in enumerate(stem_names):
        stem = sources[i].cpu().numpy()
        save_wav(outdir / f"{name}.wav", stem, sr)
        print(f"Saved {name} -> {outdir / f'{name}.wav'}")


if __name__ == "__main__":
    main()