# scripts/generate_all_csvs.py

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")


def process_csv(
    input_csv: Path,
    output_csv: Path,
    base_dir: Path,
    path_column: str = "Path",
    label_column: str = "ClassId",
) -> None:
    """Process a GTSRB-format CSV and generate a cleaner CSV with full paths."""
    df = pd.read_csv(input_csv)
    df["Filename"] = df[path_column].apply(lambda p: str(base_dir / p))
    df = df[["Filename", label_column]]
    df.to_csv(output_csv, index=False)
    logging.info(f"[+] Saved {output_csv} ({len(df)} samples)")


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_jobs = [
        ("Train.csv", "train.csv"),
        ("Test.csv", "test.csv"),
        ("Meta.csv", "meta.csv"),
    ]

    for in_file, out_file in csv_jobs:
        process_csv(
            input_csv=raw_dir / in_file,
            output_csv=processed_dir / out_file,
            base_dir=raw_dir,
        )


if __name__ == "__main__":
    main()
