import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(map(str, cmd)))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input audio file path")
    parser.add_argument("--model", default="mdx_extra", help="Demucs model name")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--ml_model", default="models/spatial_refiner.joblib", help="Path to trained ML model")
    parser.add_argument("--ml_blend", default="0.2", help="ML blend factor")
    parser.add_argument("--use_deap", action="store_true", help="Use DEAP optimisation stage")
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    track_name = infile.stem

    stems_dir = Path("outputs/stems") / args.model / track_name
    features_csv = Path("outputs/features") / f"{track_name}_features.csv"
    spatial_ml_csv = Path("outputs/spatial") / f"{track_name}_spatial_ml.csv"
    spatial_deap_csv = Path("outputs/spatial") / f"{track_name}_spatial_deap.csv"
    render_outdir = Path("outputs/rendered") / track_name
    final_mix_name = f"{track_name}_enhanced.wav"

    py = sys.executable

    run_command([
        py, "src/separate_demucs.py",
        "--infile", str(infile),
        "--outdir", "outputs/stems",
        "--model", args.model,
        "--device", args.device,
    ])

    run_command([
        py, "src/extract_features.py",
        "--stemdir", str(stems_dir),
        "--outfile", str(features_csv),
    ])

    run_command([
        py, "src/spatial_decision.py",
        "--features", str(features_csv),
        "--outfile", str(spatial_ml_csv),
        "--track", track_name,
        "--model", args.ml_model,
        "--ml_blend", str(args.ml_blend),
    ])

    spatial_csv_to_render = spatial_ml_csv

    if args.use_deap:
        run_command([
            py, "src/deap_optimise.py",
            "--features", str(features_csv),
            "--spatial", str(spatial_ml_csv),
            "--outfile", str(spatial_deap_csv),
        ])
        spatial_csv_to_render = spatial_deap_csv

    run_command([
        py, "src/render_spatial_mix.py",
        "--stemdir", str(stems_dir),
        "--spatial", str(spatial_csv_to_render),
        "--outdir", str(render_outdir),
        "--mixname", final_mix_name,
    ])


    print("\nDone.")
    print(f"Final mix: {render_outdir / final_mix_name}")


if __name__ == "__main__":
    main()