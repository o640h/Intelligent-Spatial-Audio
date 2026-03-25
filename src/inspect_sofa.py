from pathlib import Path
import argparse
import sofar as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa", required=True, help="Path to .sofa file")
    args = parser.parse_args()

    sofa_path = Path(args.sofa)
    sofa = sf.read_sofa(sofa_path)

    print("\n=== BASIC INFO ===")
    print("GLOBAL_SOFAConventions:", getattr(sofa, "GLOBAL_SOFAConventions", None))
    print("GLOBAL_DataType:", getattr(sofa, "GLOBAL_DataType", None))
    print("Data_IR shape:", getattr(sofa, "Data_IR", None).shape if hasattr(sofa, "Data_IR") else None)
    print("SourcePosition shape:", getattr(sofa, "SourcePosition", None).shape if hasattr(sofa, "SourcePosition") else None)
    print("ReceiverPosition shape:", getattr(sofa, "ReceiverPosition", None).shape if hasattr(sofa, "ReceiverPosition") else None)
    print("Data_SamplingRate:", getattr(sofa, "Data_SamplingRate", None))

    print("\n=== AVAILABLE ATTRIBUTES ===")
    for key in dir(sofa):
        if key.startswith("_"):
            continue
        try:
            val = getattr(sofa, key)
            shape = getattr(val, "shape", None)
            if shape is not None:
                print(f"{key}: shape={shape}")
        except Exception:
            pass


if __name__ == "__main__":
    main()