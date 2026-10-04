from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsRegressor

from .data import LATLON_COORD_COLUMNS, DatasetBundle, add_polar_coords, fill_env_features, load_krill_dataset
from .model import AutoregressiveGPModel, ClimateConditionedGPModel, FishCastModel, LatentStateAugmentedGPModel


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise ModuleNotFoundError(
            "matplotlib is required for FishCast visualizations. Install it in the active environment first."
        ) from exc
    return plt, Ellipse


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def _normalize_column(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (values - stats["mean"]) / stats["std"]


def _denormalize_column(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (values * stats["std"]) + stats["mean"]


def spatial_kernel_covariances_latlon(
    model: ClimateConditionedGPModel,
    bundle: DatasetBundle,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Learned spatial kernel covariance at each point, as (N, 2, 2) in [lat, lon] degrees.

    Polar-coordinate runs learn the kernel on the POLAR_X/POLAR_Y plane; the covariance is
    mapped back to local (lat, lon) with the projection's Jacobian so plots stay comparable.
    """
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    spatial_columns = list(getattr(bundle, "coord_columns", LATLON_COORD_COLUMNS))[1:]
    points = pd.DataFrame({"LATITUDE": latitudes, "LONGITUDE": longitudes})
    polar = "POLAR_X" in spatial_columns
    if polar:
        points = add_polar_coords(points)
    normalized = np.column_stack(
        [_normalize_column(points[column].to_numpy(), bundle.coord_stats[column]) for column in spatial_columns]
    ).astype(np.float32)
    base_kernel = model.gp_model.covar_module.base_kernel
    with torch.no_grad():
        sigma_norm = base_kernel.covariance_network(torch.tensor(normalized)).detach().cpu().numpy().astype(np.float64)
    scale = np.diag([bundle.coord_stats[column]["std"] for column in spatial_columns])
    sigma = scale @ sigma_norm @ scale
    if not polar:
        return sigma
    lam = np.deg2rad(longitudes)
    colat = 90.0 + latitudes
    k = np.pi / 180.0
    jacobian = np.empty((len(lam), 2, 2))
    jacobian[:, 0, 0] = np.cos(lam)
    jacobian[:, 0, 1] = -colat * k * np.sin(lam)
    jacobian[:, 1, 0] = np.sin(lam)
    jacobian[:, 1, 1] = colat * k * np.cos(lam)
    inverse = np.linalg.inv(jacobian)
    return inverse @ sigma @ np.transpose(inverse, (0, 2, 1))


def load_trained_fishcast(
    run_dir: str | Path,
    data_path: str | Path = "data/krillcast_merged.csv",
    device: str = "cpu",
) -> Tuple[ClimateConditionedGPModel, DatasetBundle, Dict[str, object]]:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    checkpoint = torch.load(run_dir / "fishcast_model.pt", map_location="cpu")
    model_name = metrics.get("model_name", "climate_conditioned_gp")
    if model_name == "autoregressive_sequence":
        raise ValueError(
            "load_trained_fishcast currently supports the GP family only. "
            "Use the autoregressive model with a dedicated sequence-forecast script."
        )

    bundle = load_krill_dataset(
        csv_path=data_path,
        num_inducing_points=int(metrics["data"]["num_inducing_points"]),
        include_lag_features=model_name in {"autoregressive_gp", "lag_feature_gp", "latent_state_augmented_gp"},
        lag_density_step=int(metrics["config"].get("lag_density_step", 1)),
        lag_mean_window=int(metrics["config"].get("lag_mean_window", 3)),
        lag_hotspot_step=int(metrics["config"].get("lag_hotspot_step", 1)),
        # Runs saved before preprocessing v2 have none of these keys and load with the original pipeline.
        preprocessing_version=int(metrics["data"].get("preprocessing_version", 1)),
        spatial_coords=metrics["data"].get("spatial_coords", "latlon"),
        test_fraction=float(metrics["data"].get("test_fraction", 0.0)),
        validation_fraction=float(metrics["data"].get("validation_fraction", 0.15)),
        hotspot_quantile=float(metrics["data"].get("hotspot_quantile", 0.9)),
        lat_sector_step=float(metrics["data"].get("lat_sector_step", 5.0)),
        lon_sector_step=float(metrics["data"].get("lon_sector_step", 5.0)),
    )
    model_class: type[ClimateConditionedGPModel]
    model_kwargs: Dict[str, object] = {
        "num_features": bundle.features.shape[1],
        "inducing_points": bundle.inducing_points,
        "hidden_dim": int(metrics["config"]["hidden_dim"]),
        "kernel_jitter": float(metrics["config"]["kernel_jitter"]),
    }
    if model_name in {"autoregressive_gp", "lag_feature_gp"}:
        model_class = AutoregressiveGPModel
    elif model_name == "latent_state_augmented_gp":
        model_class = LatentStateAugmentedGPModel
        model_kwargs.update(
            {
                "env_feature_count": len([column for column in bundle.feature_columns if not column.startswith("KRILL_") and not column.startswith("HOTSPOT_")]),
                "lag_feature_count": len([column for column in bundle.feature_columns if column.startswith("KRILL_") or column.startswith("HOTSPOT_")]),
                "latent_state_dim": int(metrics["config"].get("latent_state_dim", 8)),
            }
        )
    else:
        model_class = ClimateConditionedGPModel

    model = model_class(**model_kwargs)
    model.load_state_dict(_clean_state_dict_keys(checkpoint["model"]))
    model.likelihood.load_state_dict(_clean_state_dict_keys(checkpoint["likelihood"]))
    model.to(device)
    model.eval()
    model.gp_model.eval()
    model.likelihood.eval()
    return model, bundle, metrics


def build_forecast_grid(
    bundle: DatasetBundle,
    target_year: float | None = None,
    lat_points: int = 120,
    lon_points: int = 160,
    year_window: float = 2.0,
    neighbors: int = 12,
) -> pd.DataFrame:
    frame = bundle.dataframe.copy()
    if target_year is None:
        target_year = float(frame["FRACTIONAL_YEAR"].median())

    lat_min, lat_max = frame["LATITUDE"].min(), frame["LATITUDE"].max()
    lon_min, lon_max = frame["LONGITUDE"].min(), frame["LONGITUDE"].max()
    latitudes = np.linspace(lat_min, lat_max, lat_points)
    longitudes = np.linspace(lon_min, lon_max, lon_points)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    year_mask = np.abs(frame["FRACTIONAL_YEAR"] - target_year) <= year_window
    candidate = frame.loc[year_mask].copy()
    if len(candidate) < max(neighbors, 32):
        candidate = frame.iloc[
            np.argsort(np.abs(frame["FRACTIONAL_YEAR"].to_numpy() - target_year))[: max(256, neighbors)]
        ].copy()

    candidate = fill_env_features(candidate, bundle.feature_columns, bundle.feature_stats)

    coords = candidate[["LATITUDE", "LONGITUDE"]].to_numpy()
    prediction_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    n_neighbors = min(neighbors, len(candidate))
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    regressor.fit(coords, candidate[bundle.feature_columns].to_numpy())
    estimated_features = regressor.predict(prediction_points)

    grid = pd.DataFrame(
        {
            "FRACTIONAL_YEAR": target_year,
            "LATITUDE": prediction_points[:, 0],
            "LONGITUDE": prediction_points[:, 1],
        }
    )
    for index, column in enumerate(bundle.feature_columns):
        grid[column] = estimated_features[:, index]
    return grid


def predict_grid(
    model: FishCastModel,
    bundle: DatasetBundle,
    grid: pd.DataFrame,
    device: str = "cpu",
) -> pd.DataFrame:
    feature_array = np.column_stack(
        [
            _normalize_column(grid[column].to_numpy(), bundle.feature_stats[column])
            for column in bundle.feature_columns
        ]
    ).astype(np.float32)
    coord_columns = getattr(bundle, "coord_columns", LATLON_COORD_COLUMNS)
    coord_source = add_polar_coords(grid) if "POLAR_X" in coord_columns else grid
    coord_array = np.column_stack(
        [
            _normalize_column(coord_source[column].to_numpy(), bundle.coord_stats[column])
            for column in coord_columns
        ]
    ).astype(np.float32)

    features = torch.tensor(feature_array, dtype=torch.float32, device=device)
    coords = torch.tensor(coord_array, dtype=torch.float32, device=device)

    with torch.no_grad(), torch.inference_mode():
        outputs = model(features, coords)
        latent_mean = outputs["latent"].detach().cpu().numpy().ravel()
        density_log = outputs["density"].detach().cpu().numpy().ravel()
        hotspot_prob = outputs["hotspot_probability"].detach().cpu().numpy().ravel()

    result = grid.copy()
    result["latent_mean"] = latent_mean
    result["predicted_log_density"] = density_log
    result["predicted_density"] = np.expm1(density_log).clip(min=0.0)
    result["hotspot_probability"] = hotspot_prob
    return result


def _save_forecast_plot(frame: pd.DataFrame, value_column: str, title: str, output_path: Path) -> None:
    plt, _ = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 7))
    latitudes = np.sort(frame["LATITUDE"].unique())
    longitudes = np.sort(frame["LONGITUDE"].unique())
    pivot = frame.pivot(index="LATITUDE", columns="LONGITUDE", values=value_column).sort_index()
    mesh = ax.pcolormesh(longitudes, latitudes, pivot.to_numpy(), shading="auto")
    fig.colorbar(mesh, ax=ax, label=value_column.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_density_change_frame(
    baseline_predictions: pd.DataFrame,
    target_predictions: pd.DataFrame,
    baseline_year: float,
    target_year: float,
) -> pd.DataFrame:
    baseline_density_column = f"predicted_density_{baseline_year:.2f}"
    target_density_column = f"predicted_density_{target_year:.2f}"
    baseline_frame = baseline_predictions[["LATITUDE", "LONGITUDE", "predicted_density"]].rename(
        columns={"predicted_density": baseline_density_column}
    )
    target_frame = target_predictions.copy().rename(columns={"predicted_density": target_density_column})
    result = target_frame.merge(baseline_frame, on=["LATITUDE", "LONGITUDE"], how="inner", validate="one_to_one")
    result["historical_density"] = result[baseline_density_column]
    result["density_change"] = result[target_density_column] - result[baseline_density_column]
    return result


def _save_density_change_plot(
    frame: pd.DataFrame,
    baseline_year: float,
    target_year: float,
    output_path: Path,
) -> None:
    plt, _ = _require_matplotlib()
    from matplotlib import colors

    fig, ax = plt.subplots(figsize=(11, 7))
    latitudes = np.sort(frame["LATITUDE"].unique())
    longitudes = np.sort(frame["LONGITUDE"].unique())
    pivot = frame.pivot(index="LATITUDE", columns="LONGITUDE", values="density_change").sort_index()
    values = pivot.to_numpy()
    max_abs_change = float(np.nanmax(np.abs(values))) if values.size else 0.0
    zero_mask_threshold = max(max_abs_change * 0.01, 1e-6)
    masked_values = np.ma.masked_where(np.abs(values) <= zero_mask_threshold, values)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    if max_abs_change <= 0.0:
        max_abs_change = 1e-6
    norm = colors.TwoSlopeNorm(vmin=-max_abs_change, vcenter=0.0, vmax=max_abs_change)
    mesh = ax.pcolormesh(longitudes, latitudes, masked_values, shading="auto", cmap=cmap, norm=norm)
    fig.colorbar(mesh, ax=ax, label="Predicted Density Change")
    ax.set_facecolor("white")
    ax.set_title(f"FishCast Krill Density Change ({baseline_year:.2f} to {target_year:.2f})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_forecast_visualizations(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    target_year: float,
    baseline_predictions: pd.DataFrame | None = None,
    baseline_year: float | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "forecast_grid_predictions.csv", index=False)
    _save_forecast_plot(
        predictions,
        value_column="predicted_density",
        title=f"FishCast Predicted Krill Density ({target_year:.2f})",
        output_path=output_dir / "forecast_density.png",
    )
    _save_forecast_plot(
        predictions,
        value_column="hotspot_probability",
        title=f"FishCast Hotspot Probability ({target_year:.2f})",
        output_path=output_dir / "forecast_hotspot_probability.png",
    )
    if baseline_predictions is not None and baseline_year is not None:
        density_change_frame = _build_density_change_frame(
            baseline_predictions=baseline_predictions,
            target_predictions=predictions,
            baseline_year=baseline_year,
            target_year=target_year,
        )
        density_change_frame.to_csv(output_dir / "forecast_density_change.csv", index=False)
        _save_density_change_plot(
            frame=density_change_frame,
            baseline_year=baseline_year,
            target_year=target_year,
            output_path=output_dir / "forecast_density_change.png",
        )


def save_kernel_visualization(
    model: FishCastModel,
    bundle: DatasetBundle,
    output_dir: str | Path,
    lat_points: int = 18,
    lon_points: int = 24,
    ellipse_scale: float = 2.5,
) -> None:
    plt, Ellipse = _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = bundle.dataframe
    latitudes = np.linspace(frame["LATITUDE"].min(), frame["LATITUDE"].max(), lat_points)
    longitudes = np.linspace(frame["LONGITUDE"].min(), frame["LONGITUDE"].max(), lon_points)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    covariance_matrices = spatial_kernel_covariances_latlon(model, bundle, lat_grid.ravel(), lon_grid.ravel())

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(frame["LONGITUDE"], frame["LATITUDE"], s=3, alpha=0.08, color="0.45")

    ellipse_rows = []
    for latitude, longitude, sigma_raw in zip(lat_grid.ravel(), lon_grid.ravel(), covariance_matrices):
        sigma_plot = np.array(
            [
                [sigma_raw[1, 1], sigma_raw[1, 0]],
                [sigma_raw[0, 1], sigma_raw[0, 0]],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_plot)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        width = ellipse_scale * 2.0 * np.sqrt(max(eigenvalues[0], 1e-9))
        height = ellipse_scale * 2.0 * np.sqrt(max(eigenvalues[1], 1e-9))
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        ellipse = Ellipse(
            xy=(longitude, latitude),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linewidth=0.8,
            alpha=0.8,
            edgecolor="#d94841",
        )
        ax.add_patch(ellipse)
        ellipse_rows.append(
            {
                "LATITUDE": latitude,
                "LONGITUDE": longitude,
                "ellipse_width_degrees": width,
                "ellipse_height_degrees": height,
                "ellipse_angle_degrees": angle,
            }
        )

    ax.set_title("FishCast Learned Spatial Kernel Geometry")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(frame["LONGITUDE"].min(), frame["LONGITUDE"].max())
    ax.set_ylim(frame["LATITUDE"].min(), frame["LATITUDE"].max())
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_geometry.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(ellipse_rows).to_csv(output_dir / "kernel_ellipses.csv", index=False)
