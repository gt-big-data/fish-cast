from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans


DEFAULT_ENV_FEATURES = ["SST", "SIA", "SIM", "SIT", "SSS", "SAP", "SLP"]
LATLON_COORD_COLUMNS = ["FRACTIONAL_YEAR", "LATITUDE", "LONGITUDE"]
POLAR_COORD_COLUMNS = ["FRACTIONAL_YEAR", "POLAR_X", "POLAR_Y"]
COORD_COLUMNS = POLAR_COORD_COLUMNS
TARGET_COLUMN = "NUMBER_OF_KRILL_UNDER_1M2"
MISSING_SUFFIX = "_MISSING"
# 1 = original pipeline (kept so runs trained before this change still load).
# 2 = train-only fitting, held-out test split, sea-ice rule, missing flags,
#     aligned lag features, seam-free polar coordinates.
PREPROCESSING_VERSION = 2
@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    features: torch.Tensor
    coords: torch.Tensor
    target: torch.Tensor
    hotspot: torch.Tensor
    inducing_points: torch.Tensor
    feature_columns: List[str]
    feature_stats: Dict[str, Dict[str, float]]
    coord_stats: Dict[str, Dict[str, float]]
    target_stats: Dict[str, float]
    metadata: Dict[str, object]
    test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    coord_columns: List[str] = field(default_factory=lambda: list(LATLON_COORD_COLUMNS))


@dataclass
class SequenceDatasetBundle:
    dataframe: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    past_inputs: torch.Tensor
    future_env: torch.Tensor
    future_target: torch.Tensor
    future_hotspot: torch.Tensor
    feature_columns: List[str]
    feature_stats: Dict[str, Dict[str, float]]
    target_stats: Dict[str, float]
    metadata: Dict[str, object]


@dataclass
class GridSequenceDatasetBundle:
    dataframe: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    past_inputs: torch.Tensor
    future_env: torch.Tensor
    future_target: torch.Tensor
    future_hotspot: torch.Tensor
    future_mask: torch.Tensor
    feature_columns: List[str]
    feature_stats: Dict[str, Dict[str, float]]
    target_stats: Dict[str, float]
    metadata: Dict[str, object]


def _normalize_frame(
    frame: pd.DataFrame,
    columns: List[str],
    fit_index: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Median-fill and z-score `columns`.

    Statistics come from the rows in `fit_index` (positional) when given, so validation
    and test rows never influence the scaling. Missing-value indicator columns are left as 0/1.
    """
    normalized = frame.copy()
    stats: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column.endswith(MISSING_SUFFIX):
            stats[column] = {"median": 0.0, "mean": 0.0, "std": 1.0}
            continue
        fit_values = normalized[column] if fit_index is None else normalized[column].iloc[fit_index]
        median = float(fit_values.median()) if fit_values.notna().any() else 0.0
        normalized[column] = normalized[column].fillna(median)
        fit_values = fit_values.fillna(median)
        mean = float(fit_values.mean())
        std = float(fit_values.std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        normalized[column] = (normalized[column] - mean) / std
        stats[column] = {"median": median, "mean": mean, "std": std}
    return normalized, stats


def _build_sector_hotspots(
    frame: pd.DataFrame,
    quantile: float,
    lat_step: float,
    lon_step: float,
    fit_index: np.ndarray | None = None,
    require_positive: bool = False,
    min_sector_hauls: int = 0,
) -> pd.Series:
    """Label a haul a hotspot if it reaches its sector's `quantile` of krill density.

    With `fit_index`, each sector's threshold is learned from training rows only; sectors
    with no training rows (or fewer than `min_sector_hauls`) use the overall training quantile.

    `require_positive`: 36% of hauls catch no krill, so in many sectors the 90th percentile
    is 0 and ">= threshold" labelled empty hauls as hotspots (1,915 labels, 609 of them
    zero-krill). With it set, a hotspot must also contain krill.
    """
    sector_key = build_sector_keys(frame, lat_step=lat_step, lon_step=lon_step)
    if fit_index is None:
        thresholds = frame.groupby(sector_key)[TARGET_COLUMN].transform(lambda s: s.quantile(quantile))
    else:
        fit_frame = frame.iloc[fit_index]
        fit_keys = sector_key.iloc[fit_index]
        per_sector = fit_frame.groupby(fit_keys)[TARGET_COLUMN].quantile(quantile)
        if min_sector_hauls > 0:
            counts = fit_keys.value_counts()
            per_sector = per_sector[counts.reindex(per_sector.index).fillna(0) >= min_sector_hauls]
        fallback = float(fit_frame[TARGET_COLUMN].quantile(quantile))
        thresholds = sector_key.map(per_sector).fillna(fallback)
    hotspot = frame[TARGET_COLUMN] >= thresholds
    if require_positive:
        hotspot &= frame[TARGET_COLUMN] > 0
    return hotspot.astype(np.float32)


def add_polar_coords(frame: pd.DataFrame) -> pd.DataFrame:
    """Add POLAR_X / POLAR_Y: a south-polar azimuthal projection, in degrees from the pole.

    Raw longitude jumps from +180 to -180, so a kernel on (lat, lon) treats neighbouring
    hauls either side of the antimeridian as 360 degrees apart. KRILLBASE covers the full
    circle, so the spatial kernel works on this seam-free plane instead.
    """
    out = frame.copy()
    colatitude = 90.0 + out["LATITUDE"].astype("float64")
    radians = np.deg2rad(out["LONGITUDE"].astype("float64"))
    out["POLAR_X"] = colatitude * np.cos(radians)
    out["POLAR_Y"] = colatitude * np.sin(radians)
    return out


def apply_sea_ice_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """Sea-ice thickness is blank where there is no sea ice; record it as 0, not missing.

    In KRILLBASE 11,942 of 12,544 hauls have SIT blank with SIA == 0. Median-filling them
    told the model there was ice of median thickness at ice-free stations.
    """
    if "SIT" not in frame.columns or "SIA" not in frame.columns:
        return frame
    out = frame.copy()
    ice_free = out["SIT"].isna() & (out["SIA"] == 0)
    out.loc[ice_free, "SIT"] = 0.0
    return out


def missing_indicator_columns(frame: pd.DataFrame, columns: List[str], fit_index: np.ndarray) -> List[str]:
    """One `<col>_MISSING` flag per distinct missingness pattern present in the training rows."""
    names: List[str] = []
    seen: List[np.ndarray] = []
    for column in columns:
        mask = frame[column].isna().to_numpy()
        if not mask[fit_index].any():
            continue
        if any(np.array_equal(mask, other) for other in seen):
            continue
        seen.append(mask)
        names.append(f"{column}{MISSING_SUFFIX}")
    return names


def add_missing_indicators(frame: pd.DataFrame, indicator_columns: List[str]) -> pd.DataFrame:
    out = frame.copy()
    for name in indicator_columns:
        base = name[: -len(MISSING_SUFFIX)]
        out[name] = out[base].isna().astype(np.float32) if base in out.columns else np.float32(1.0)
    return out


def fill_env_features(
    frame: pd.DataFrame,
    feature_columns: List[str],
    feature_stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Inference-time preprocessing that mirrors training for a new grid (e.g. CMIP6).

    Runs trained with preprocessing v2 carry `_MISSING` columns; for those the sea-ice rule
    and missing flags are applied before median filling. v1 runs get the old median fill only.
    """
    indicators = [column for column in feature_columns if column.endswith(MISSING_SUFFIX)]
    out = frame
    if indicators:
        out = apply_sea_ice_rules(out)
        out = add_missing_indicators(out, indicators)
    else:
        out = out.copy()
    for column in feature_columns:
        if column in indicators:
            continue
        if column not in out.columns:
            out[column] = feature_stats[column]["median"]
        out[column] = out[column].fillna(feature_stats[column]["median"])
    return out


def build_sector_keys(frame: pd.DataFrame, lat_step: float, lon_step: float) -> pd.Series:
    sector_lat = np.floor(frame["LATITUDE"] / lat_step) * lat_step
    sector_lon = np.floor(frame["LONGITUDE"] / lon_step) * lon_step
    return sector_lat.round(3).astype(str) + "_" + sector_lon.round(3).astype(str)


def lag_feature_column_names(lag_density_step: int, lag_mean_window: int, lag_hotspot_step: int) -> List[str]:
    return [
        f"KRILL_LAG{lag_density_step}_LOG_DENSITY",
        f"KRILL_LAG{lag_mean_window}_LOG_MEAN",
        f"HOTSPOT_LAG{lag_hotspot_step}",
    ]


def _add_lag_features(
    frame: pd.DataFrame,
    lat_step: float,
    lon_step: float,
    lag_density_step: int,
    lag_mean_window: int,
    lag_hotspot_step: int,
    fit_index: np.ndarray | None = None,
    preserve_order: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    enriched = frame.copy()
    enriched["_row_order"] = np.arange(len(enriched))
    enriched["sector_key"] = build_sector_keys(enriched, lat_step=lat_step, lon_step=lon_step)
    enriched["target_log"] = np.log1p(enriched[TARGET_COLUMN].astype(np.float32))
    enriched = enriched.sort_values(["sector_key", "FRACTIONAL_YEAR", "DATE"]).reset_index(drop=True)

    grouped = enriched.groupby("sector_key", group_keys=False)
    lag_columns = lag_feature_column_names(lag_density_step, lag_mean_window, lag_hotspot_step)
    density_column, mean_column, hotspot_column = lag_columns
    enriched[density_column] = grouped["target_log"].shift(lag_density_step)
    enriched[mean_column] = grouped["target_log"].transform(
        lambda s: s.shift(1).rolling(window=lag_mean_window, min_periods=1).mean()
    )
    enriched[hotspot_column] = grouped["hotspot"].shift(lag_hotspot_step)

    fit_rows = np.ones(len(enriched), dtype=bool) if fit_index is None else enriched["_row_order"].isin(fit_index).to_numpy()
    for column in lag_columns:
        source = enriched.loc[fit_rows, column]
        median = float(source.median()) if source.notna().any() else 0.0
        enriched[column] = enriched[column].fillna(median)
    if preserve_order:
        # The sort above reorders rows by sector; restore the input order so the lag
        # features line up with coords, targets and split indices built from `frame`.
        enriched = enriched.sort_values("_row_order").reset_index(drop=True)
    return enriched.drop(columns=["_row_order"]), lag_columns


def _split_indices(
    frame: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split: oldest hauls train, then validation, newest held out as test.

    The test rows are never used for fitting, threshold tuning or checkpoint selection.
    """
    years = frame["FRACTIONAL_YEAR"]
    if test_fraction > 0.0:
        test_cutoff = years.quantile(1.0 - test_fraction)
        val_cutoff = years.quantile(1.0 - test_fraction - validation_fraction)
        train_mask = years < val_cutoff
        val_mask = (years >= val_cutoff) & (years < test_cutoff)
        test_mask = years >= test_cutoff
    else:
        cutoff = years.quantile(1.0 - validation_fraction)
        train_mask = years < cutoff
        val_mask = ~train_mask
        test_mask = pd.Series(False, index=years.index)
    return (
        np.flatnonzero(train_mask.to_numpy()),
        np.flatnonzero(val_mask.to_numpy()),
        np.flatnonzero(test_mask.to_numpy()),
    )


def load_krill_dataset(
    csv_path: str | Path,
    num_inducing_points: int = 500,
    hotspot_quantile: float = 0.9,
    validation_fraction: float = 0.15,
    lat_sector_step: float = 5.0,
    lon_sector_step: float = 5.0,
    log_target: bool = True,
    include_lag_features: bool = False,
    lag_density_step: int = 1,
    lag_mean_window: int = 3,
    lag_hotspot_step: int = 1,
    test_fraction: float = 0.15,
    spatial_coords: str = "polar",
    preprocessing_version: int = PREPROCESSING_VERSION,
) -> DatasetBundle:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    frame["DATE"] = pd.to_datetime(frame["DATE"])

    legacy = preprocessing_version < 2
    if legacy:
        # Reproduce the original pipeline exactly so pre-fix runs still load.
        test_fraction = 0.0
        spatial_coords = "latlon"
    if spatial_coords not in {"polar", "latlon"}:
        raise ValueError("spatial_coords must be 'polar' or 'latlon'.")

    env_columns = [column for column in DEFAULT_ENV_FEATURES if column in frame.columns]
    if not env_columns:
        raise ValueError("No environmental covariates were found in the merged CSV.")

    train_indices, val_indices, test_indices = _split_indices(
        frame, validation_fraction=validation_fraction, test_fraction=test_fraction
    )
    fit_index = None if legacy else train_indices

    if not legacy:
        frame = apply_sea_ice_rules(frame)

    frame["hotspot"] = _build_sector_hotspots(
        frame,
        quantile=hotspot_quantile,
        lat_step=lat_sector_step,
        lon_step=lon_sector_step,
        fit_index=fit_index,
        require_positive=not legacy,
        min_sector_hauls=0 if legacy else 20,
    )

    indicator_columns = [] if legacy else missing_indicator_columns(frame, env_columns, train_indices)
    if indicator_columns:
        frame = add_missing_indicators(frame, indicator_columns)
    feature_columns = env_columns + indicator_columns

    if include_lag_features:
        working_frame, lag_columns = _add_lag_features(
            frame,
            lat_step=lat_sector_step,
            lon_step=lon_sector_step,
            lag_density_step=lag_density_step,
            lag_mean_window=lag_mean_window,
            lag_hotspot_step=lag_hotspot_step,
            fit_index=fit_index,
            preserve_order=not legacy,
        )
        feature_columns = feature_columns + lag_columns
        if not legacy:
            aligned = (
                np.array_equal(working_frame["DATE"].to_numpy(), frame["DATE"].to_numpy())
                and np.allclose(working_frame["LATITUDE"].to_numpy(), frame["LATITUDE"].to_numpy())
                and np.allclose(working_frame["LONGITUDE"].to_numpy(), frame["LONGITUDE"].to_numpy())
            )
            if not aligned:
                raise RuntimeError("Lag features are not row-aligned with coordinates and targets.")
    else:
        working_frame = frame.copy()
        lag_columns = []

    normalized_features, feature_stats = _normalize_frame(working_frame, feature_columns, fit_index=fit_index)

    coord_columns = list(POLAR_COORD_COLUMNS if spatial_coords == "polar" else LATLON_COORD_COLUMNS)
    coord_frame = add_polar_coords(frame) if spatial_coords == "polar" else frame
    normalized_coords, coord_stats = _normalize_frame(coord_frame, coord_columns, fit_index=fit_index)

    target_values = frame[TARGET_COLUMN].astype(np.float32).to_numpy()
    transformed_target = np.log1p(target_values) if log_target else target_values

    coord_array = normalized_coords[coord_columns].to_numpy()
    kmeans_input = coord_array if legacy else coord_array[train_indices]
    num_clusters = min(num_inducing_points, len(kmeans_input))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        inducing_array = kmeans.fit(kmeans_input).cluster_centers_.astype(np.float32)

    hotspot_values = frame["hotspot"].to_numpy()
    metadata = {
        "num_rows": len(frame),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "test_rows": len(test_indices),
        "train_years": [float(frame["FRACTIONAL_YEAR"].iloc[train_indices].min()), float(frame["FRACTIONAL_YEAR"].iloc[train_indices].max())] if len(train_indices) else None,
        "val_years": [float(frame["FRACTIONAL_YEAR"].iloc[val_indices].min()), float(frame["FRACTIONAL_YEAR"].iloc[val_indices].max())] if len(val_indices) else None,
        "test_years": [float(frame["FRACTIONAL_YEAR"].iloc[test_indices].min()), float(frame["FRACTIONAL_YEAR"].iloc[test_indices].max())] if len(test_indices) else None,
        "hotspot_rate": float(hotspot_values.mean()),
        "hotspot_rate_train": float(hotspot_values[train_indices].mean()) if len(train_indices) else float("nan"),
        "hotspot_rate_test": float(hotspot_values[test_indices].mean()) if len(test_indices) else None,
        "target_quantile_90": float(frame[TARGET_COLUMN].quantile(0.9)),
        "log_target": log_target,
        "num_inducing_points": num_clusters,
        "lag_density_step": lag_density_step if include_lag_features else None,
        "lag_mean_window": lag_mean_window if include_lag_features else None,
        "lag_hotspot_step": lag_hotspot_step if include_lag_features else None,
        "preprocessing_version": 1 if legacy else PREPROCESSING_VERSION,
        "spatial_coords": spatial_coords,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "hotspot_quantile": hotspot_quantile,
        "lat_sector_step": lat_sector_step,
        "lon_sector_step": lon_sector_step,
        "missing_indicator_columns": indicator_columns,
    }

    fit_target = target_values if legacy else target_values[train_indices]
    fit_transformed = transformed_target if legacy else transformed_target[train_indices]
    target_stats = {
        "mean_raw": float(fit_target.mean()),
        "std_raw": float(fit_target.std()),
        "mean_transformed": float(np.mean(fit_transformed)),
        "std_transformed": float(np.std(fit_transformed)),
    }

    return DatasetBundle(
        dataframe=working_frame,
        train_indices=train_indices,
        val_indices=val_indices,
        features=torch.tensor(normalized_features[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
        coords=torch.tensor(coord_array.astype(np.float32), dtype=torch.float32),
        target=torch.tensor(transformed_target, dtype=torch.float32).unsqueeze(-1),
        hotspot=torch.tensor(hotspot_values, dtype=torch.float32).unsqueeze(-1),
        inducing_points=torch.tensor(inducing_array, dtype=torch.float32),
        feature_columns=feature_columns,
        feature_stats=feature_stats,
        coord_stats=coord_stats,
        target_stats=target_stats,
        metadata=metadata,
        test_indices=test_indices,
        coord_columns=coord_columns,
    )


def load_autoregressive_dataset(
    csv_path: str | Path,
    seq_len: int = 5,
    horizon_len: int = 3,
    validation_fraction: float = 0.15,
    hotspot_quantile: float = 0.9,
    lat_sector_step: float = 5.0,
    lon_sector_step: float = 5.0,
    log_target: bool = True,
) -> SequenceDatasetBundle:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    feature_columns = [column for column in DEFAULT_ENV_FEATURES if column in frame.columns]
    if not feature_columns:
        raise ValueError("No environmental covariates were found in the merged CSV.")
    frame = apply_sea_ice_rules(frame)

    frame["hotspot"] = _build_sector_hotspots(
        frame,
        quantile=hotspot_quantile,
        lat_step=lat_sector_step,
        lon_step=lon_sector_step,
    )
    frame["sector_key"] = build_sector_keys(frame, lat_step=lat_sector_step, lon_step=lon_sector_step)
    frame["YEAR_INT"] = frame["FRACTIONAL_YEAR"].round().astype(int)
    frame["target_log"] = np.log1p(frame[TARGET_COLUMN].astype(np.float32)) if log_target else frame[TARGET_COLUMN].astype(np.float32)

    aggregate_map = {column: "mean" for column in feature_columns}
    aggregate_map.update(
        {
            "LATITUDE": "mean",
            "LONGITUDE": "mean",
            "FRACTIONAL_YEAR": "mean",
            "target_log": "mean",
            "hotspot": "max",
        }
    )
    yearly = (
        frame.groupby(["sector_key", "YEAR_INT"], as_index=False)
        .agg(aggregate_map)
        .sort_values(["sector_key", "YEAR_INT"])
        .reset_index(drop=True)
    )

    normalized_features, feature_stats = _normalize_frame(yearly, feature_columns)
    yearly_normalized = yearly.copy()
    for column in feature_columns:
        yearly_normalized[column] = normalized_features[column]

    sequence_rows = []
    past_sequences = []
    future_env_sequences = []
    future_target_sequences = []
    future_hotspot_sequences = []

    for sector_key, sector_frame in yearly_normalized.groupby("sector_key"):
        sector_frame = sector_frame.sort_values("YEAR_INT").reset_index(drop=True)
        if len(sector_frame) < seq_len + horizon_len:
            continue

        env_values = sector_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = sector_frame["target_log"].to_numpy(dtype=np.float32)
        hotspot_values = sector_frame["hotspot"].to_numpy(dtype=np.float32)
        year_values = sector_frame["YEAR_INT"].to_numpy(dtype=np.int32)

        for start in range(0, len(sector_frame) - seq_len - horizon_len + 1):
            past_slice = slice(start, start + seq_len)
            future_slice = slice(start + seq_len, start + seq_len + horizon_len)
            past_input = np.concatenate(
                [
                    env_values[past_slice],
                    target_values[past_slice, None],
                    hotspot_values[past_slice, None],
                ],
                axis=1,
            )
            future_env = env_values[future_slice]
            future_target = target_values[future_slice]
            future_hotspot = hotspot_values[future_slice]

            past_sequences.append(past_input)
            future_env_sequences.append(future_env)
            future_target_sequences.append(future_target)
            future_hotspot_sequences.append(future_hotspot)
            sequence_rows.append(
                {
                    "sector_key": sector_key,
                    "forecast_start_year": int(year_values[start + seq_len]),
                    "forecast_end_year": int(year_values[start + seq_len + horizon_len - 1]),
                }
            )

    if not past_sequences:
        raise ValueError("No valid autoregressive windows could be constructed. Reduce seq_len or horizon_len.")

    sequence_frame = pd.DataFrame(sequence_rows)
    cutoff = sequence_frame["forecast_start_year"].quantile(1.0 - validation_fraction)
    train_indices = np.flatnonzero((sequence_frame["forecast_start_year"] < cutoff).to_numpy())
    val_indices = np.flatnonzero((sequence_frame["forecast_start_year"] >= cutoff).to_numpy())

    target_concat = np.concatenate(future_target_sequences)
    target_stats = {
        "mean_transformed": float(np.mean(target_concat)),
        "std_transformed": float(np.std(target_concat)),
    }
    metadata = {
        "num_sequences": len(past_sequences),
        "train_sequences": len(train_indices),
        "val_sequences": len(val_indices),
        "seq_len": seq_len,
        "horizon_len": horizon_len,
        "num_sectors": int(yearly["sector_key"].nunique()),
        "log_target": log_target,
    }

    return SequenceDatasetBundle(
        dataframe=sequence_frame,
        train_indices=train_indices,
        val_indices=val_indices,
        past_inputs=torch.tensor(np.stack(past_sequences), dtype=torch.float32),
        future_env=torch.tensor(np.stack(future_env_sequences), dtype=torch.float32),
        future_target=torch.tensor(np.stack(future_target_sequences), dtype=torch.float32),
        future_hotspot=torch.tensor(np.stack(future_hotspot_sequences), dtype=torch.float32),
        feature_columns=feature_columns,
        feature_stats=feature_stats,
        target_stats=target_stats,
        metadata=metadata,
    )


def load_convlstm_dataset(
    csv_path: str | Path,
    seq_len: int = 5,
    horizon_len: int = 3,
    validation_fraction: float = 0.15,
    hotspot_quantile: float = 0.9,
    lat_sector_step: float = 5.0,
    lon_sector_step: float = 5.0,
    log_target: bool = True,
) -> GridSequenceDatasetBundle:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    feature_columns = [column for column in DEFAULT_ENV_FEATURES if column in frame.columns]
    if not feature_columns:
        raise ValueError("No environmental covariates were found in the merged CSV.")
    frame = apply_sea_ice_rules(frame)

    frame["hotspot"] = _build_sector_hotspots(
        frame,
        quantile=hotspot_quantile,
        lat_step=lat_sector_step,
        lon_step=lon_sector_step,
    )
    frame["sector_lat"] = np.floor(frame["LATITUDE"] / lat_sector_step) * lat_sector_step
    frame["sector_lon"] = np.floor(frame["LONGITUDE"] / lon_sector_step) * lon_sector_step
    frame["sector_key"] = build_sector_keys(frame, lat_step=lat_sector_step, lon_step=lon_sector_step)
    frame["YEAR_INT"] = frame["FRACTIONAL_YEAR"].round().astype(int)
    frame["target_log"] = np.log1p(frame[TARGET_COLUMN].astype(np.float32)) if log_target else frame[TARGET_COLUMN].astype(np.float32)

    aggregate_map = {column: "mean" for column in feature_columns}
    aggregate_map.update(
        {
            "LATITUDE": "mean",
            "LONGITUDE": "mean",
            "sector_lat": "first",
            "sector_lon": "first",
            "FRACTIONAL_YEAR": "mean",
            "target_log": "mean",
            "hotspot": "max",
        }
    )
    yearly = (
        frame.groupby(["YEAR_INT", "sector_key"], as_index=False)
        .agg(aggregate_map)
        .sort_values(["YEAR_INT", "sector_lat", "sector_lon"])
        .reset_index(drop=True)
    )

    normalized_features, feature_stats = _normalize_frame(yearly, feature_columns)
    yearly_normalized = yearly.copy()
    for column in feature_columns:
        yearly_normalized[column] = normalized_features[column]

    lat_levels = np.sort(yearly["sector_lat"].unique())
    lon_levels = np.sort(yearly["sector_lon"].unique())
    lat_to_idx = {value: index for index, value in enumerate(lat_levels)}
    lon_to_idx = {value: index for index, value in enumerate(lon_levels)}
    years = np.sort(yearly_normalized["YEAR_INT"].unique())

    env_grids = []
    target_grids = []
    hotspot_grids = []
    mask_grids = []
    for year in years:
        year_frame = yearly_normalized.loc[yearly_normalized["YEAR_INT"] == year]
        env_grid = np.zeros((len(feature_columns), len(lat_levels), len(lon_levels)), dtype=np.float32)
        target_grid = np.zeros((1, len(lat_levels), len(lon_levels)), dtype=np.float32)
        hotspot_grid = np.zeros((1, len(lat_levels), len(lon_levels)), dtype=np.float32)
        mask_grid = np.zeros((1, len(lat_levels), len(lon_levels)), dtype=np.float32)

        for row in year_frame.itertuples(index=False):
            lat_idx = lat_to_idx[float(row.sector_lat)]
            lon_idx = lon_to_idx[float(row.sector_lon)]
            env_grid[:, lat_idx, lon_idx] = np.asarray([getattr(row, column) for column in feature_columns], dtype=np.float32)
            target_grid[0, lat_idx, lon_idx] = float(row.target_log)
            hotspot_grid[0, lat_idx, lon_idx] = float(row.hotspot)
            mask_grid[0, lat_idx, lon_idx] = 1.0

        env_grids.append(env_grid)
        target_grids.append(target_grid)
        hotspot_grids.append(hotspot_grid)
        mask_grids.append(mask_grid)

    env_array = np.stack(env_grids)
    target_array = np.stack(target_grids)
    hotspot_array = np.stack(hotspot_grids)
    mask_array = np.stack(mask_grids)

    sequence_rows = []
    past_sequences = []
    future_env_sequences = []
    future_target_sequences = []
    future_hotspot_sequences = []
    future_mask_sequences = []
    for start in range(0, len(years) - seq_len - horizon_len + 1):
        past_slice = slice(start, start + seq_len)
        future_slice = slice(start + seq_len, start + seq_len + horizon_len)
        past_input = np.concatenate(
            [
                env_array[past_slice],
                target_array[past_slice],
                hotspot_array[past_slice],
            ],
            axis=1,
        )
        future_env = env_array[future_slice]
        future_target = target_array[future_slice]
        future_hotspot = hotspot_array[future_slice]
        future_mask = mask_array[future_slice]

        past_sequences.append(past_input)
        future_env_sequences.append(future_env)
        future_target_sequences.append(future_target)
        future_hotspot_sequences.append(future_hotspot)
        future_mask_sequences.append(future_mask)
        sequence_rows.append(
            {
                "forecast_start_year": int(years[start + seq_len]),
                "forecast_end_year": int(years[start + seq_len + horizon_len - 1]),
            }
        )

    if not past_sequences:
        raise ValueError("No valid ConvLSTM windows could be constructed. Reduce seq_len or horizon_len.")

    sequence_frame = pd.DataFrame(sequence_rows)
    cutoff = sequence_frame["forecast_start_year"].quantile(1.0 - validation_fraction)
    train_indices = np.flatnonzero((sequence_frame["forecast_start_year"] < cutoff).to_numpy())
    val_indices = np.flatnonzero((sequence_frame["forecast_start_year"] >= cutoff).to_numpy())

    target_concat = np.concatenate([x.reshape(-1) for x in future_target_sequences])
    mask_concat = np.concatenate([x.reshape(-1) for x in future_mask_sequences]).astype(bool)
    observed_targets = target_concat[mask_concat]
    target_stats = {
        "mean_transformed": float(np.mean(observed_targets)),
        "std_transformed": float(np.std(observed_targets)),
    }
    metadata = {
        "num_sequences": len(past_sequences),
        "train_sequences": len(train_indices),
        "val_sequences": len(val_indices),
        "seq_len": seq_len,
        "horizon_len": horizon_len,
        "num_years": int(len(years)),
        "grid_height": int(len(lat_levels)),
        "grid_width": int(len(lon_levels)),
        "lat_sector_step": lat_sector_step,
        "lon_sector_step": lon_sector_step,
        "log_target": log_target,
    }

    return GridSequenceDatasetBundle(
        dataframe=sequence_frame,
        train_indices=train_indices,
        val_indices=val_indices,
        past_inputs=torch.tensor(np.stack(past_sequences), dtype=torch.float32),
        future_env=torch.tensor(np.stack(future_env_sequences), dtype=torch.float32),
        future_target=torch.tensor(np.stack(future_target_sequences), dtype=torch.float32),
        future_hotspot=torch.tensor(np.stack(future_hotspot_sequences), dtype=torch.float32),
        future_mask=torch.tensor(np.stack(future_mask_sequences), dtype=torch.float32),
        feature_columns=feature_columns,
        feature_stats=feature_stats,
        target_stats=target_stats,
        metadata=metadata,
    )
