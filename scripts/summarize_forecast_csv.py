from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize predicted_density statistics for one or more FishCast forecast CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more forecast CSV files containing a predicted_density column.",
    )
    return parser.parse_args()


def summarize_csv(path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(path, usecols=["predicted_density"])
    series = df["predicted_density"].dropna()
    if series.empty:
        raise ValueError(f"{path} has no non-null predicted_density values.")

    return {
        "file": str(path),
        "rows": int(len(series)),
        "min": float(series.min()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def main() -> None:
    args = parse_args()
    for raw_path in args.csv_files:
        path = Path(raw_path)
        stats = summarize_csv(path)
        print(path)
        print(f"  rows:   {stats['rows']:,}")
        print(f"  min:    {stats['min']:.6f}")
        print(f"  mean:   {stats['mean']:.6f}")
        print(f"  median: {stats['median']:.6f}")
        print(f"  max:    {stats['max']:.6f}")


if __name__ == "__main__":
    main()
