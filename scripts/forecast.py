from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from fishcast.data import build_sector_keys, fill_env_features, lag_feature_column_names
from fishcast.visualization import load_trained_fishcast, predict_grid


DEFAULT_VARIABLE_MAPPING = {
    "lat": "LATITUDE",
    "latitude": "LATITUDE",
    "nav_lat": "LATITUDE",
    "lon": "LONGITUDE",
    "longitude": "LONGITUDE",
    "nav_lon": "LONGITUDE",
    "tos": "SST",
    "sst": "SST",
    "siconc": "SIA",
    "siconca": "SIA",
    "sim": "SIM",
    "sitimean": "SIM",
    "sithick": "SIT",
    "sit": "SIT",
    "sos": "SSS",
    "sss": "SSS",
    "ps": "SAP",
    "surface_pressure": "SAP",
    "psl": "SLP",
    "sea_level_pressure": "SLP",
}


def _require_xarray():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise ModuleNotFoundError(
            "xarray is required for forecast.py. Install xarray (and a NetCDF backend like netCDF4 or h5netcdf) "
            "in the active environment before running forecasts."
        ) from exc
    return xr


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise ModuleNotFoundError(
            "matplotlib is required for timelapse frame rendering. Install matplotlib in the active environment."
        ) from exc
    return plt


def _parse_mapping(raw_items: list[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_VARIABLE_MAPPING)
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --map value '{item}'. Use source=TARGET, for example tos=SST.")
        source, target = item.split("=", 1)
        mapping[source.strip()] = target.strip()
    return mapping


def _resolve_run_dir(run_dir: str | Path, objective: str = "auc") -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "metrics.json").exists() and (run_dir / "fishcast_model.pt").exists():
        return run_dir

    sweep_summary = run_dir / "sweep_summary.json"
    if sweep_summary.exists():
        summary = json.loads(sweep_summary.read_text())
        key = "best_by_auc" if objective == "auc" else "best_by_rmse"
        return Path(summary[key]["output_dir"])

    raise FileNotFoundError(
        f"Could not resolve a trained run directory from {run_dir}. "
        "Pass either a concrete run directory with metrics.json/fishcast_model.pt or a sweep directory with sweep_summary.json."
    )


def _get_coord_name(dataset, candidates: list[str], axis_name: str) -> str:
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims or candidate in dataset.variables:
            return candidate
    raise KeyError(f"Could not find a {axis_name} coordinate. Tried: {', '.join(candidates)}")


def _target_timestamp_from_year(target_year: float) -> str:
    year = int(target_year)
    fraction = target_year - year
    month = min(max(int(round(fraction * 12)) + 1, 1), 12)
    return f"{year}-{month:02d}-15"


def _format_year_label(target_year: float) -> str:
    if float(target_year).is_integer():
        return str(int(target_year))
    return f"{target_year:.2f}".replace(".", "_")


TIME_COORD_CANDIDATES = ("time", "Date", "date", "valid_time", "Time", "TIME", "t", "month")


def _get_time_coord_name(ds) -> str | None:
    """Return the name of the time-like coordinate/dimension, or None.

    The regridded CMIP6 files produced by the merge notebook name this axis
    ``Date`` rather than ``time``. Hardcoding ``"time"`` silently skipped time
    selection and emitted every monthly slice for a single requested year.
    """
    import numpy as _np

    for name in TIME_COORD_CANDIDATES:
        if name in ds.coords or name in ds.dims:
            if name in ds.coords and not _np.issubdtype(ds[name].dtype, _np.datetime64):
                continue
            return name
    for name, coord in ds.coords.items():
        if _np.issubdtype(coord.dtype, _np.datetime64) and coord.ndim == 1:
            return str(name)
    return None


def _select_time_slice(ds, target_year: float):
    """Select the single time step nearest ``target_year``. Returns (dataset, label)."""
    time_name = _get_time_coord_name(ds)
    if time_name is None:
        max_dims = max((ds[v].ndim for v in ds.data_vars), default=0)
        if max_dims > 2:
            raise ValueError(
                "No time-like coordinate was found in the NetCDF file, but its data variables "
                f"are {max_dims}-dimensional. Rename the time axis to one of {TIME_COORD_CANDIDATES} "
                "so a single forecast slice can be selected."
            )
        return ds, f"{target_year:.2f}"

    target_timestamp = _target_timestamp_from_year(target_year)
    sub_ds = ds.sel({time_name: target_timestamp}, method="nearest")
    selected = sub_ds[time_name].values
    label = str(selected)

    try:
        import numpy as _np

        requested = _np.datetime64(target_timestamp)
        gap_days = abs((_np.datetime64(selected) - requested) / _np.timedelta64(1, "D"))
        if gap_days > 366:
            print(
                f"  [warn] requested {target_timestamp} but nearest available slice is {label} "
                f"({gap_days:.0f} days away) - the climate file does not cover this year."
            )
        requested_month = int(str(requested)[5:7])
        selected_month = int(str(_np.datetime64(selected))[5:7])
        if requested_month != selected_month:
            print(
                f"  [warn] requested month {requested_month:02d} but got {selected_month:02d} ({label}). "
                "Sea ice and SST swing seasonally, so this slice is not comparable to the others - "
                "the climate file likely ends before the requested year."
            )
    except Exception:
        pass
    return sub_ds, label


def _wrap_longitudes(df):
    """Convert 0..360 longitudes to the -180..180 convention used by the training data.

    Coordinates are z-scored with the *training* statistics, so a 0..360 grid pushes
    the entire Pacific sector far outside the range the model ever saw.
    """
    if "LONGITUDE" not in df.columns:
        return df
    lon = df["LONGITUDE"].astype("float64")
    if lon.max() > 180.0:
        df = df.copy()
        df["LONGITUDE"] = ((lon + 180.0) % 360.0) - 180.0
    return df


def _apply_ocean_mask(df, mask_column: str):
    """Drop land / ice-shelf cells using the climate model's own missing-value mask.

    The CMIP6 ocean fields are NaN over land. Median-filling them (the previous
    behaviour) invented plausible-looking SST and salinity over the Antarctic
    continent and the model duly predicted krill there.
    """
    if not mask_column or mask_column not in df.columns:
        return df, 0
    before = len(df)
    df = df[df[mask_column].notna()].reset_index(drop=True)
    return df, before - len(df)


def _assert_unique_grid(df, context: str):
    """Fail loudly if more than one row exists per (LATITUDE, LONGITUDE)."""
    duplicated = df.duplicated(subset=["LATITUDE", "LONGITUDE"]).sum()
    if duplicated:
        unique_cells = len(df) - duplicated
        factor = len(df) / unique_cells if unique_cells else float("nan")
        raise ValueError(
            f"{context}: expected one row per grid cell but got {len(df)} rows for "
            f"{unique_cells} unique cells (~{factor:.0f}x duplication). An extra dimension "
            "survived time selection - check the NetCDF coordinate names."
        )



def _parse_year_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _build_rollout_years(start_year: float, end_year: float, step_years: int) -> List[float]:
    if step_years < 1:
        raise ValueError("frame_step_years must be >= 1.")
    if start_year > end_year:
        raise ValueError("start_year must be <= target_year.")
    start_int = int(math.floor(start_year))
    end_int = int(math.floor(end_year))
    years = list(range(start_int, end_int + 1))
    rollout_years = [float(year) for year in years]
    if not math.isclose(end_year, float(end_int), rel_tol=0.0, abs_tol=1e-6):
        rollout_years.append(float(end_year))
    return rollout_years


def load_and_prep_cmip6(
    nc_path: str | Path,
    target_year: float,
    feature_columns: list[str],
    feature_stats: Dict[str, Dict[str, float]],
    variable_mapping: Dict[str, str],
    lat_cutoff: float = -40.0,
    ocean_mask_variable: str = "SST",
) -> pd.DataFrame:
    xr = _require_xarray()
    ds = xr.open_dataset(nc_path)

    lat_name = _get_coord_name(ds, ["lat", "latitude", "nav_lat"], "latitude")
    lon_name = _get_coord_name(ds, ["lon", "longitude", "nav_lon"], "longitude")

    if lat_name in ds.coords:
        ds = ds.sortby(lat_name)
        ds = ds.sel({lat_name: slice(-90, lat_cutoff)})

    ds, _time_label = _select_time_slice(ds, target_year)

    reverse_mapping = {target: source for source, target in variable_mapping.items()}
    wanted_targets = ["LATITUDE", "LONGITUDE", *feature_columns]
    wanted_sources = [reverse_mapping[target] for target in wanted_targets if target in reverse_mapping]
    available_sources = [column for column in wanted_sources if column in ds.variables or column in ds.coords]
    if available_sources:
        ds = ds[available_sources]

    df = ds.to_dataframe().reset_index()
    df = df.rename(columns=variable_mapping)
    df = _wrap_longitudes(df)

    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        raise ValueError(
            "The NetCDF file did not resolve to LATITUDE/LONGITUDE columns after renaming. "
            "Use --map to supply the correct coordinate mappings."
        )

    df["FRACTIONAL_YEAR"] = target_year

    df, _dropped = _apply_ocean_mask(df, ocean_mask_variable)

    df = fill_env_features(df, feature_columns, feature_stats)

    keep_columns = ["FRACTIONAL_YEAR", "LATITUDE", "LONGITUDE", *feature_columns]
    df = df[keep_columns].dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    _assert_unique_grid(df, f"load_and_prep_cmip6({target_year})")
    for column in keep_columns:
        df[column] = df[column].astype(np.float32)
    return df


def _prepare_cmip6_dataframe(
    sub_ds,
    ds,
    target_year: float,
    feature_columns: list[str],
    feature_stats: Dict[str, Dict[str, float]],
    variable_mapping: Dict[str, str],
    ocean_mask_variable: str = "SST",
) -> pd.DataFrame:
    reverse_mapping = {target: source for source, target in variable_mapping.items()}
    env_columns = [c for c in feature_columns if not c.startswith("KRILL_") and not c.startswith("HOTSPOT_")]
    wanted_targets = ["LATITUDE", "LONGITUDE", *env_columns]
    wanted_sources = [reverse_mapping[target] for target in wanted_targets if target in reverse_mapping]
    available_sources = [column for column in wanted_sources if column in sub_ds.variables or column in sub_ds.coords]
    if available_sources:
        sub_ds = sub_ds[available_sources]

    df = sub_ds.to_dataframe().reset_index().rename(columns=variable_mapping)
    df = _wrap_longitudes(df)
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        raise ValueError(
            "The NetCDF file did not resolve to LATITUDE/LONGITUDE columns after renaming. "
            "Use --map to supply the correct coordinate mappings."
        )
    df["FRACTIONAL_YEAR"] = np.float32(target_year)

    df, _dropped = _apply_ocean_mask(df, ocean_mask_variable)

    df = fill_env_features(df, env_columns, feature_stats)

    keep_columns = ["FRACTIONAL_YEAR", "LATITUDE", "LONGITUDE", *env_columns]
    df = df[keep_columns].dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    _assert_unique_grid(df, f"cmip6 slice for {target_year}")
    for column in keep_columns:
        df[column] = df[column].astype(np.float32)
    return df


def load_cmip6_slices_for_years(
    nc_path: str | Path,
    years: Sequence[float],
    feature_columns: list[str],
    feature_stats: Dict[str, Dict[str, float]],
    variable_mapping: Dict[str, str],
    lat_cutoff: float = -40.0,
    ocean_mask_variable: str = "SST",
) -> List[Tuple[str, pd.DataFrame]]:
    xr = _require_xarray()
    ds = xr.open_dataset(nc_path)

    lat_name = _get_coord_name(ds, ["lat", "latitude", "nav_lat"], "latitude")
    if lat_name in ds.coords:
        ds = ds.sortby(lat_name)
        ds = ds.sel({lat_name: slice(-90, lat_cutoff)})

    slices = []
    for year in years:
        sub_ds, label = _select_time_slice(ds, year)
        df = _prepare_cmip6_dataframe(
            sub_ds=sub_ds,
            ds=ds,
            target_year=year,
            feature_columns=feature_columns,
            feature_stats=feature_stats,
            variable_mapping=variable_mapping,
            ocean_mask_variable=ocean_mask_variable,
        )
        slices.append((label, df))
    return slices


def build_sector_histories(bundle: object, lat_sector_step: float, lon_sector_step: float) -> Dict[str, Dict[str, List[float]]]:
    frame = bundle.dataframe.copy()
    if "target_log" not in frame.columns:
        if "NUMBER_OF_KRILL_UNDER_1M2" not in frame.columns:
            raise ValueError("Training dataframe does not contain krill density needed for autoregressive rollout.")
        frame["target_log"] = np.log1p(frame["NUMBER_OF_KRILL_UNDER_1M2"].astype(np.float32))
    if "hotspot" not in frame.columns:
        raise ValueError("Training dataframe does not contain hotspot values needed for autoregressive rollout.")
    frame["sector_key"] = build_sector_keys(frame, lat_step=lat_sector_step, lon_step=lon_sector_step)
    frame = frame.sort_values(["sector_key", "FRACTIONAL_YEAR", "DATE"]).reset_index(drop=True)

    histories: Dict[str, Dict[str, List[float]]] = {}
    for sector_key, sector_frame in frame.groupby("sector_key"):
        histories[sector_key] = {
            "density_log": sector_frame["target_log"].astype(float).tolist(),
            "hotspot": sector_frame["hotspot"].astype(float).tolist(),
        }
    return histories


def add_autoregressive_features(
    grid: pd.DataFrame,
    bundle,
    histories: Dict[str, Dict[str, List[float]]],
    lag_density_step: int,
    lag_mean_window: int,
    lag_hotspot_step: int,
    lat_sector_step: float,
    lon_sector_step: float,
) -> pd.DataFrame:
    lag_columns = lag_feature_column_names(lag_density_step, lag_mean_window, lag_hotspot_step)
    density_column, mean_column, hotspot_column = lag_columns
    result = grid.copy()
    result["sector_key"] = build_sector_keys(result, lat_step=lat_sector_step, lon_step=lon_sector_step)

    density_values = []
    mean_values = []
    hotspot_values = []
    density_median = bundle.feature_stats[density_column]["median"]
    mean_median = bundle.feature_stats[mean_column]["median"]
    hotspot_median = bundle.feature_stats[hotspot_column]["median"]

    for sector_key in result["sector_key"]:
        history = histories.get(sector_key, {"density_log": [], "hotspot": []})
        density_history = history["density_log"]
        hotspot_history = history["hotspot"]

        density_values.append(density_history[-lag_density_step] if len(density_history) >= lag_density_step else density_median)
        mean_values.append(float(np.mean(density_history[-lag_mean_window:])) if density_history else mean_median)
        hotspot_values.append(hotspot_history[-lag_hotspot_step] if len(hotspot_history) >= lag_hotspot_step else hotspot_median)

    result[density_column] = np.asarray(density_values, dtype=np.float32)
    result[mean_column] = np.asarray(mean_values, dtype=np.float32)
    result[hotspot_column] = np.asarray(hotspot_values, dtype=np.float32)
    return result.drop(columns=["sector_key"])


def update_sector_histories_from_predictions(
    predictions: pd.DataFrame,
    histories: Dict[str, Dict[str, List[float]]],
    threshold: float,
    lat_sector_step: float,
    lon_sector_step: float,
) -> None:
    working = predictions.copy()
    working["sector_key"] = build_sector_keys(working, lat_step=lat_sector_step, lon_step=lon_sector_step)
    grouped = working.groupby("sector_key", as_index=False).agg(
        {"predicted_log_density": "mean", "hotspot_probability": "mean"}
    )
    for row in grouped.itertuples(index=False):
        history = histories.setdefault(row.sector_key, {"density_log": [], "hotspot": []})
        history["density_log"].append(float(row.predicted_log_density))
        history["hotspot"].append(float(row.hotspot_probability >= threshold))


def predict_grid_batched(
    model,
    bundle,
    grid: pd.DataFrame,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    total_rows = len(grid)
    starts = range(0, total_rows, batch_size)
    progress = tqdm(starts, total=(total_rows + batch_size - 1) // batch_size, desc="Forecast batches")
    collected = []

    for start in progress:
        stop = min(start + batch_size, total_rows)
        batch = grid.iloc[start:stop].copy()
        predictions = predict_grid(model, bundle, batch, device=device)
        progress.set_postfix(rows=f"{stop:,}/{total_rows:,}")
        collected.append(predictions)

    return pd.concat(collected, ignore_index=True)


def _aggregate_prediction_cells(frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in ["latent_mean", "predicted_log_density", "predicted_density", "hotspot_probability"]
        if column in frame.columns
    ]
    passthrough_columns = [
        column
        for column in frame.columns
        if column not in {"LATITUDE", "LONGITUDE", *value_columns}
    ]
    if not frame.duplicated(subset=["LATITUDE", "LONGITUDE"]).any():
        return frame

    aggregated = (
        frame.groupby(["LATITUDE", "LONGITUDE"], as_index=False)[value_columns]
        .mean()
        .sort_values(["LATITUDE", "LONGITUDE"])
        .reset_index(drop=True)
    )
    for column in passthrough_columns:
        aggregated[column] = frame[column].iloc[0]
    ordered_columns = [
        column
        for column in frame.columns
        if column in aggregated.columns
    ]
    return aggregated[ordered_columns]


def _save_forecast_plot(frame: pd.DataFrame, value_column: str, title: str, output_path: Path) -> None:
    plt = _require_matplotlib()
    plotting_frame = _aggregate_prediction_cells(frame)
    fig, ax = plt.subplots(figsize=(11, 7))
    latitudes = np.sort(plotting_frame["LATITUDE"].unique())
    longitudes = np.sort(plotting_frame["LONGITUDE"].unique())
    pivot = plotting_frame.pivot(index="LATITUDE", columns="LONGITUDE", values=value_column).sort_index()
    mesh = ax.pcolormesh(longitudes, latitudes, pivot.to_numpy(), shading="auto")
    fig.colorbar(mesh, ax=ax, label=value_column.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_frame_data(predictions: pd.DataFrame, output_path: Path, output_format: str) -> None:
    if output_format == "csv":
        predictions.to_csv(output_path, index=False)
        return
    if output_format == "parquet":
        predictions.to_parquet(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {output_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FishCast forecasts on a CMIP6-style NetCDF file.")
    parser.add_argument(
        "--run-dir",
        default="artifacts/FishCast_Test_v2",
        help="Concrete trained run directory, or a sweep directory containing sweep_summary.json.",
    )
    parser.add_argument(
        "--objective",
        choices=["auc", "rmse"],
        default="auc",
        help="If --run-dir is a sweep directory, choose the best run by this objective.",
    )
    parser.add_argument(
        "--train-data",
        default="data/krillcast_merged.csv",
        help="Training CSV used to rebuild feature normalizers.",
    )
    parser.add_argument(
        "--cmip6-file",
        default="data/mpi_esm1_2_ssp245_regridded.nc",
        help="Path to the NetCDF file used for forecasting.",
    )
    parser.add_argument("--target-year", type=float, default=2050.0, help="Fractional year to forecast.")
    parser.add_argument("--start-year", type=float, default=None, help="First rollout year. Defaults to the year after the last observed training year.")
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of grid cells to forecast per batch to control memory usage.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path. Defaults to <resolved-run-dir>/forecast_results_<year>.csv",
    )
    parser.add_argument(
        "--timelapse-dir",
        default="",
        help="Optional directory for timelapse frames and selected per-year outputs. Defaults to <resolved-run-dir>/timelapse_to_<target-year>.",
    )
    parser.add_argument(
        "--lat-cutoff",
        type=float,
        default=-40.0,
        help="Only keep grid cells south of this latitude.",
    )
    parser.add_argument(
        "--ocean-mask-variable",
        default="SST",
        help="Drop grid cells where this variable is missing in the climate file (land / ice-shelf mask). "
             "Pass an empty string to disable masking.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Extra source=TARGET column mappings, for example --map chl=CHL.",
    )
    parser.add_argument("--rollout-steps", type=int, default=1, help="Legacy option for contiguous future slices. Ignored when yearly timelapse rollout is active.")
    parser.add_argument("--frame-step-years", type=int, default=1, help="Render/export one frame every N rollout years.")
    parser.add_argument("--export-years", default="", help="Optional comma-separated years to render/export instead of using --frame-step-years.")
    parser.add_argument("--write-frame-data", action="store_true", help="Write per-frame forecast data files in addition to PNGs.")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="csv", help="Data format used when --write-frame-data is enabled.")
    parser.add_argument("--lat-sector-step", type=float, default=5.0, help="Sector latitude step used for autoregressive GP-style lag features.")
    parser.add_argument("--lon-sector-step", type=float, default=5.0, help="Sector longitude step used for autoregressive GP-style lag features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_run_dir = _resolve_run_dir(args.run_dir, objective=args.objective)

    print(f"Loading FishCast model from {resolved_run_dir}...")
    model, bundle, metrics = load_trained_fishcast(
        run_dir=resolved_run_dir,
        data_path=args.train_data,
        device=args.device,
    )

    variable_mapping = _parse_mapping(args.map)
    start_year = (
        float(args.start_year)
        if args.start_year is not None
        else float(math.floor(bundle.dataframe["FRACTIONAL_YEAR"].max()) + 1)
    )
    rollout_years = _build_rollout_years(start_year, args.target_year, args.frame_step_years)
    if args.export_years.strip():
        export_years = {float(year) for year in _parse_year_list(args.export_years)}
    else:
        export_years = {year for index, year in enumerate(rollout_years) if (index % args.frame_step_years) == 0}
    export_years.add(float(args.target_year))

    slices = load_cmip6_slices_for_years(
        nc_path=args.cmip6_file,
        years=rollout_years,
        feature_columns=bundle.feature_columns,
        feature_stats=bundle.feature_stats,
        variable_mapping=variable_mapping,
        lat_cutoff=args.lat_cutoff,
        ocean_mask_variable=args.ocean_mask_variable,
    )

    model_name = metrics.get("model_name", "climate_conditioned_gp")
    histories = None
    if model_name in {"autoregressive_gp", "lag_feature_gp", "latent_state_augmented_gp"}:
        histories = build_sector_histories(bundle, args.lat_sector_step, args.lon_sector_step)
        lag_density_step = int(metrics["config"].get("lag_density_step", 1))
        lag_mean_window = int(metrics["config"].get("lag_mean_window", 3))
        lag_hotspot_step = int(metrics["config"].get("lag_hotspot_step", 1))
        threshold = float(metrics.get("selected_validation_threshold", 0.5))

    timelapse_dir = (
        Path(args.timelapse_dir)
        if args.timelapse_dir
        else resolved_run_dir / f"timelapse_to_{_format_year_label(args.target_year)}"
    )
    frame_dir = timelapse_dir / "frames"
    data_dir = timelapse_dir / "frame_data"
    timelapse_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    if args.write_frame_data:
        data_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else resolved_run_dir / f"forecast_results_{args.target_year:.1f}.csv"
    if output_csv.exists():
        output_csv.unlink()

    total_rows = 0
    rendered_frames = []
    for step_index, ((label, forecast_grid_df), rollout_year) in enumerate(zip(slices, rollout_years), start=1):
        working_grid = forecast_grid_df
        if histories is not None:
            working_grid = add_autoregressive_features(
                forecast_grid_df,
                bundle,
                histories,
                lag_density_step=lag_density_step,
                lag_mean_window=lag_mean_window,
                lag_hotspot_step=lag_hotspot_step,
                lat_sector_step=args.lat_sector_step,
                lon_sector_step=args.lon_sector_step,
            )
        print(
            f"Generating forecasts for rollout year {rollout_year:.0f} "
            f"({step_index}/{len(slices)}) with {len(working_grid)} grid cells..."
        )
        predictions = predict_grid_batched(
            model=model,
            bundle=bundle,
            grid=working_grid,
            device=args.device,
            batch_size=args.batch_size,
        )
        predictions.insert(0, "forecast_slice", label)
        predictions.insert(1, "forecast_year", rollout_year)

        should_export_frame = any(
            math.isclose(float(rollout_year), float(export_year), rel_tol=0.0, abs_tol=1e-6)
            for export_year in export_years
        )
        if should_export_frame:
            year_label = _format_year_label(rollout_year)
            density_png = frame_dir / f"forecast_density_{year_label}.png"
            hotspot_png = frame_dir / f"forecast_hotspot_probability_{year_label}.png"
            _save_forecast_plot(
                predictions,
                value_column="predicted_density",
                title=f"FishCast Predicted Krill Density ({rollout_year:.0f})",
                output_path=density_png,
            )
            _save_forecast_plot(
                predictions,
                value_column="hotspot_probability",
                title=f"FishCast Hotspot Probability ({rollout_year:.0f})",
                output_path=hotspot_png,
            )
            rendered_frames.append(
                {
                    "forecast_year": rollout_year,
                    "forecast_slice": label,
                    "density_png": str(density_png),
                    "hotspot_png": str(hotspot_png),
                }
            )
            if args.write_frame_data:
                extension = "csv" if args.output_format == "csv" else "parquet"
                frame_data_path = data_dir / f"forecast_{year_label}.{extension}"
                _write_frame_data(predictions, frame_data_path, args.output_format)

        if math.isclose(float(rollout_year), float(args.target_year), rel_tol=0.0, abs_tol=1e-6):
            predictions.to_csv(output_csv, index=False)
            total_rows = len(predictions)

        if histories is not None:
            update_sector_histories_from_predictions(
                predictions,
                histories,
                threshold=threshold,
                lat_sector_step=args.lat_sector_step,
                lon_sector_step=args.lon_sector_step,
            )

    manifest = {
        "model_source": str(resolved_run_dir),
        "run_model_name": model_name,
        "start_year": start_year,
        "target_year": args.target_year,
        "rollout_years": rollout_years,
        "export_years": sorted(export_years),
        "frame_step_years": args.frame_step_years,
        "frames": rendered_frames,
        "final_year_csv": str(output_csv),
        "write_frame_data": args.write_frame_data,
        "output_format": args.output_format if args.write_frame_data else None,
    }
    (timelapse_dir / "timelapse_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Done. Final-year forecasts saved to {output_csv} ({total_rows:,} rows)")
    print(f"Timelapse frames saved to {frame_dir}")
    if args.write_frame_data:
        print(f"Per-frame forecast data saved to {data_dir}")
    print(f"Model source: {resolved_run_dir}")
    print(f"Best validation AUC: {metrics.get('best_validation_auc', 'n/a')}")


if __name__ == "__main__":
    main()
