from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd

from interactive_forecast_globe import (
    _filter_to_supported_cells,
    _load_frame_table,
    _load_mpa_features,
    _load_support_points,
    build_interactive_globe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an animated FishCast anomaly globe showing density change growth over time."
    )
    parser.add_argument(
        "--timelapse-dir",
        required=True,
        help="Timelapse directory produced by forecast.py with frame_data/ and timelapse_manifest.json.",
    )
    parser.add_argument(
        "--baseline-year",
        type=float,
        default=None,
        help="Baseline forecast year used as the historical reference. Defaults to the earliest available frame.",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Output HTML path. Defaults to <timelapse-dir>/interactive_anomaly_timelapse_globe.html",
    )
    parser.add_argument(
        "--title",
        default="FishCast Animated Density Change Globe",
        help="Title shown in the HTML visualization.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="Chunk size used if a frame-data CSV requires aggregation.",
    )
    parser.add_argument(
        "--density-clip-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to clip anomaly color scaling for readability.",
    )
    parser.add_argument(
        "--sample-top-n",
        type=int,
        default=12000,
        help="Maximum number of cells displayed per frame after anomaly-based ranking.",
    )
    parser.add_argument(
        "--support-csv",
        default="data/krillcast_merged.csv",
        help="Observed LATITUDE/LONGITUDE CSV used to keep only cells near the real ocean sampling footprint.",
    )
    parser.add_argument(
        "--max-distance-deg",
        type=float,
        default=3.0,
        help="Maximum distance in lat/lon degrees from the observed footprint for a cell to remain visible.",
    )
    parser.add_argument(
        "--mpa-json",
        default="data/CCAMLR_MPA.json",
        help="GeoJSON-style CCAMLR MPA polygon file to overlay on the globe.",
    )
    return parser.parse_args()


def _format_year_label(target_year: float) -> str:
    if float(target_year).is_integer():
        return str(int(target_year))
    return f"{target_year:.2f}".replace(".", "_")


def _load_timelapse_frames(timelapse_dir: Path, chunksize: int) -> Dict[str, pd.DataFrame]:
    manifest_path = timelapse_dir / "timelapse_manifest.json"
    frame_data_dir = timelapse_dir / "frame_data"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} does not exist.")
    if not frame_data_dir.exists():
        raise FileNotFoundError(
            f"{frame_data_dir} does not exist. Re-run forecast.py with `--write-frame-data --output-format parquet`."
        )

    manifest = json.loads(manifest_path.read_text())
    frame_tables: Dict[str, pd.DataFrame] = {}
    for frame in manifest.get("frames", []):
        year = float(frame["forecast_year"])
        year_label = _format_year_label(year)
        parquet_path = frame_data_dir / f"forecast_{year_label}.parquet"
        csv_path = frame_data_dir / f"forecast_{year_label}.csv"
        if parquet_path.exists():
            frame_tables[str(int(year)) if float(year).is_integer() else str(year)] = _load_frame_table(parquet_path, chunksize)
        elif csv_path.exists():
            frame_tables[str(int(year)) if float(year).is_integer() else str(year)] = _load_frame_table(csv_path, chunksize)

    if not frame_tables:
        raise FileNotFoundError(f"No frame-data files were found in {frame_data_dir}.")
    return frame_tables


def _resolve_baseline_name(frame_tables: Dict[str, pd.DataFrame], baseline_year: float | None) -> str:
    ordered_names = sorted(frame_tables.keys(), key=lambda value: float(value))
    if baseline_year is None:
        return ordered_names[0]
    for name in ordered_names:
        if math.isclose(float(name), baseline_year, rel_tol=0.0, abs_tol=1e-6):
            return name
    raise ValueError(f"Baseline year {baseline_year} was not found in the timelapse frame data.")


def _build_anomaly_frame_tables(
    frame_tables: Dict[str, pd.DataFrame],
    baseline_name: str,
) -> Dict[str, pd.DataFrame]:
    baseline_year = float(baseline_name)
    baseline_column = f"predicted_density_{baseline_year:.2f}"
    baseline_frame = frame_tables[baseline_name][["LATITUDE", "LONGITUDE", "predicted_density"]].rename(
        columns={"predicted_density": baseline_column}
    )

    anomaly_tables: Dict[str, pd.DataFrame] = {}
    for name, table in frame_tables.items():
        target_year = float(name)
        target_column = f"predicted_density_{target_year:.2f}"
        if math.isclose(target_year, baseline_year, rel_tol=0.0, abs_tol=1e-6):
            merged = table.copy().rename(columns={"predicted_density": target_column})
            merged[f"{target_column}_target"] = merged[target_column]
            merged["historical_density"] = merged[target_column]
            merged["density_change"] = 0.0
            merged[baseline_column] = merged[target_column]
        else:
            target_frame = table.copy().rename(columns={"predicted_density": target_column})
            merged = target_frame.merge(
                baseline_frame,
                on=["LATITUDE", "LONGITUDE"],
                how="inner",
                validate="one_to_one",
            )
            merged["historical_density"] = merged[baseline_column]
            merged["density_change"] = merged[target_column] - merged[baseline_column]
        anomaly_tables[name] = merged
    return anomaly_tables


def main() -> None:
    args = parse_args()
    timelapse_dir = Path(args.timelapse_dir)
    frame_tables = _load_timelapse_frames(timelapse_dir=timelapse_dir, chunksize=args.chunksize)
    baseline_name = _resolve_baseline_name(frame_tables, args.baseline_year)
    anomaly_tables = _build_anomaly_frame_tables(frame_tables=frame_tables, baseline_name=baseline_name)

    support_points = _load_support_points(Path(args.support_csv)) if args.support_csv else pd.DataFrame()
    anomaly_tables = _filter_to_supported_cells(
        frame_tables=anomaly_tables,
        support_points=support_points,
        max_distance_deg=args.max_distance_deg,
    )
    mpa_features = _load_mpa_features(Path(args.mpa_json)) if args.mpa_json else []
    output_html = (
        Path(args.output_html)
        if args.output_html
        else timelapse_dir / "interactive_anomaly_timelapse_globe.html"
    )
    title = f"{args.title} ({baseline_name} baseline)"
    build_interactive_globe(
        frame_tables=anomaly_tables,
        output_html=output_html,
        title=title,
        density_clip_quantile=args.density_clip_quantile,
        sample_top_n=args.sample_top_n,
        mpa_features=mpa_features,
    )
    print(f"Wrote animated anomaly globe to {output_html}")


if __name__ == "__main__":
    main()
