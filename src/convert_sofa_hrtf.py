from pathlib import Path
import argparse
import numpy as np
import sofar as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa", required=True, help="Path to .sofa HRTF file")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument(
        "--elevation_tol",
        type=float,
        default=5.0,
        help="Keep only measurements near 0 deg elevation (default ±5 deg)",
    )
    args = parser.parse_args()

    sofa_path = Path(args.sofa)
    out_path = Path(args.out)

    sofa = sf.read_sofa(sofa_path)

    if not hasattr(sofa, "Data_IR"):
        raise ValueError("SOFA file has no Data_IR field.")
    if not hasattr(sofa, "SourcePosition"):
        raise ValueError("SOFA file has no SourcePosition field.")

    ir = np.asarray(sofa.Data_IR, dtype=np.float32)
    src_pos = np.asarray(sofa.SourcePosition, dtype=np.float32)

    # Expected common layout:
    # Data_IR -> (M, R, N)
    # SourcePosition -> (M, C) where C is usually [azimuth, elevation, distance]
    if ir.ndim != 3:
        raise ValueError(f"Expected Data_IR to be 3D (M,R,N), got shape {ir.shape}")
    if src_pos.ndim != 2 or src_pos.shape[1] < 2:
        raise ValueError(f"Expected SourcePosition shape (M,C>=2), got {src_pos.shape}")

    # Use horizontal plane only for now
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    keep = np.abs(el) <= args.elevation_tol

    if np.sum(keep) < 3:
        raise ValueError(
            f"Not enough horizontal-plane measurements found within ±{args.elevation_tol} degrees elevation."
        )

    az = az[keep]
    ir = ir[keep]

    # Receiver dimension should be 2 for binaural data
    if ir.shape[1] < 2:
        raise ValueError(f"Expected at least 2 receivers/ears, got {ir.shape[1]}")

    # Sort by azimuth
    order = np.argsort(az)
    az = az[order]
    ir = ir[order]

    hrir_l = ir[:, 0, :]
    hrir_r = ir[:, 1, :]

    sr = getattr(sofa, "Data_SamplingRate", None)
    if sr is None:
        raise ValueError("SOFA file has no Data_SamplingRate.")
    sr = int(np.asarray(sr).squeeze())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        azimuths=az.astype(np.float32),
        hrir_l=hrir_l.astype(np.float32),
        hrir_r=hrir_r.astype(np.float32),
        sample_rate=sr,
    )

    print(f"Saved converted HRTF to: {out_path}")
    print(f"Azimuth count: {len(az)}")
    print(f"HRIR length: {hrir_l.shape[1]}")
    print(f"Sample rate: {sr}")


if __name__ == "__main__":
    main()