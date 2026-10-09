"""Criteo preprocessing entry point.

Keep transformation logic here rather than in notebooks.
"""

from pathlib import Path
import argparse
import pandas as pd


def load_criteo(path: Path) -> pd.DataFrame:
    """Load the raw Criteo attribution file."""
    # Update separator/column handling after inspecting the downloaded file.
    return pd.read_csv(path, sep="\t")


def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    """Sort customer events chronologically."""
    return df.sort_values(["uid", "timestamp"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = load_criteo(args.input)
    df = sort_events(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
