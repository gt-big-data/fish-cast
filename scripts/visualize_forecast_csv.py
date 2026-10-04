from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is required for forecast CSV visualizations. Install it in the active environment first."
        ) from exc
    return plt, LinearSegmentedColormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize a large FishCast forecast CSV."
    )
    parser.add_argument(
        "forecast_csv",
        help="Path to a FishCast forecast CSV, for example forecast_results_2050.0.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for aggregated outputs. Defaults to a sibling folder next to the CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="Chunk size used while streaming the CSV.",
    )
    parser.add_argument(
        "--hotspot-quantile",
        type=float,
        default=0.99,
        help="Quantile used to define a high-density hotspot mask on the aggregated map.",
    )
    parser.add_argument(
        "--density-clip-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to clip the density color scale for map readability.",
    )
    return parser.parse_args()


def aggregate_forecast_csv(path: Path, chunksize: int) -> pd.DataFrame:
    grouped_chunks = []
    usecols = ["LATITUDE", "LONGITUDE", "predicted_density", "predicted_log_density", "hotspot_probability"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        grouped = (
            chunk.groupby(["LATITUDE", "LONGITUDE"], as_index=False)
            .agg(
                predicted_density_sum=("predicted_density", "sum"),
                predicted_log_density_sum=("predicted_log_density", "sum"),
                hotspot_probability_sum=("hotspot_probability", "sum"),
                duplicate_count=("predicted_density", "size"),
            )
        )
        grouped_chunks.append(grouped)

    combined = pd.concat(grouped_chunks, ignore_index=True)
    aggregated = (
        combined.groupby(["LATITUDE", "LONGITUDE"], as_index=False)
        .agg(
            predicted_density_sum=("predicted_density_sum", "sum"),
            predicted_log_density_sum=("predicted_log_density_sum", "sum"),
            hotspot_probability_sum=("hotspot_probability_sum", "sum"),
            duplicate_count=("duplicate_count", "sum"),
        )
    )
    aggregated["predicted_density"] = aggregated["predicted_density_sum"] / aggregated["duplicate_count"]
    aggregated["predicted_log_density"] = (
        aggregated["predicted_log_density_sum"] / aggregated["duplicate_count"]
    )
    aggregated["hotspot_probability"] = (
        aggregated["hotspot_probability_sum"] / aggregated["duplicate_count"]
    )
    aggregated = aggregated.drop(
        columns=["predicted_density_sum", "predicted_log_density_sum", "hotspot_probability_sum"]
    )
    return aggregated.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


def compute_summary(aggregated: pd.DataFrame, hotspot_quantile: float, density_clip_quantile: float) -> dict[str, float | int | bool]:
    density = aggregated["predicted_density"].to_numpy()
    hotspot_prob = aggregated["hotspot_probability"].to_numpy()
    summary = {
        "unique_grid_cells": int(len(aggregated)),
        "duplicate_count_per_cell_min": int(aggregated["duplicate_count"].min()),
        "duplicate_count_per_cell_max": int(aggregated["duplicate_count"].max()),
        "predicted_density_min": float(np.min(density)),
        "predicted_density_mean": float(np.mean(density)),
        "predicted_density_median": float(np.median(density)),
        "predicted_density_p90": float(np.quantile(density, 0.90)),
        "predicted_density_p99": float(np.quantile(density, 0.99)),
        "predicted_density_max": float(np.max(density)),
        "predicted_density_clip_value": float(np.quantile(density, density_clip_quantile)),
        "hotspot_probability_min": float(np.min(hotspot_prob)),
        "hotspot_probability_mean": float(np.mean(hotspot_prob)),
        "hotspot_probability_max": float(np.max(hotspot_prob)),
        "hotspot_probability_dynamic_range": float(np.max(hotspot_prob) - np.min(hotspot_prob)),
        "hotspot_probability_is_nearly_constant": bool((np.max(hotspot_prob) - np.min(hotspot_prob)) < 1e-5),
        "density_hotspot_threshold": float(np.quantile(density, hotspot_quantile)),
    }
    return summary


def _grid_for_column(aggregated: pd.DataFrame, column: str):
    pivot = aggregated.pivot(index="LATITUDE", columns="LONGITUDE", values=column).sort_index()
    latitudes = pivot.index.to_numpy()
    longitudes = pivot.columns.to_numpy()
    return latitudes, longitudes, pivot.to_numpy()


def save_density_map(
    aggregated: pd.DataFrame,
    output_path: Path,
    title: str,
    clip_value: float,
) -> None:
    plt, LinearSegmentedColormap = _require_matplotlib()
    latitudes, longitudes, grid = _grid_for_column(aggregated, "predicted_density")
    display_grid = np.log10(np.clip(grid, 0.0, clip_value) + 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "krill_density",
        ["#081d58", "#225ea8", "#1d91c0", "#41b6c4", "#a1dab4", "#fee090", "#f46d43", "#d73027"],
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#f8f4ec")
    mesh = ax.pcolormesh(longitudes, latitudes, display_grid, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("log10(1 + predicted_density clipped for readability)")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(longitudes.min(), longitudes.max())
    ax.set_ylim(latitudes.min(), latitudes.max())
    ax.grid(color="#ffffff", alpha=0.08, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_hotspot_overlay_map(
    aggregated: pd.DataFrame,
    output_path: Path,
    title: str,
    clip_value: float,
    hotspot_threshold: float,
    hotspot_quantile: float,
) -> None:
    plt, LinearSegmentedColormap = _require_matplotlib()
    latitudes, longitudes, density_grid = _grid_for_column(aggregated, "predicted_density")
    _, _, hotspot_grid = _grid_for_column(
        aggregated.assign(density_hotspot=(aggregated["predicted_density"] >= hotspot_threshold).astype(float)),
        "density_hotspot",
    )
    display_grid = np.log10(np.clip(density_grid, 0.0, clip_value) + 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "krill_density_soft",
        ["#0b1f33", "#13557b", "#2d8f9d", "#b7d7c0", "#f6e7a1"],
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#f8f4ec")
    mesh = ax.pcolormesh(longitudes, latitudes, display_grid, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("log10(1 + predicted_density clipped)")

    ax.contour(
        longitudes,
        latitudes,
        hotspot_grid,
        levels=[0.5],
        colors=["#bf1d1d"],
        linewidths=1.5,
    )
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.text(
        0.01,
        0.02,
        f"Red contour = top {(1.0 - hotspot_quantile) * 100:.0f}% of grid cells by predicted density",
        transform=ax.transAxes,
        fontsize=10,
        color="#5a2a27",
        bbox={"facecolor": "#f5f1e8", "edgecolor": "none", "alpha": 0.9, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plot(
    aggregated: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plt, _ = _require_matplotlib()
    density = aggregated["predicted_density"].to_numpy()
    log_density = np.log10(density + 1.0)
    median = float(np.median(density))
    mean = float(np.mean(density))
    p90 = float(np.quantile(density, 0.90))
    p99 = float(np.quantile(density, 0.99))

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#fbf8f2")
    ax.hist(log_density, bins=80, color="#2d8f9d", edgecolor="#0b1f33", linewidth=0.4, alpha=0.9)

    lines = [
        (median, "#0b1f33", "Median"),
        (mean, "#bf1d1d", "Mean"),
        (p90, "#e07a1f", "P90"),
        (p99, "#6a4c93", "P99"),
    ]
    for value, color, label in lines:
        ax.axvline(np.log10(value + 1.0), color=color, linewidth=2.0, label=f"{label}: {value:,.2f}")

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("log10(1 + predicted_density)")
    ax.set_ylabel("Grid-cell count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_hotspot_probability_plot(
    aggregated: pd.DataFrame,
    output_path: Path,
    title: str,
    nearly_constant: bool,
) -> None:
    plt, _ = _require_matplotlib()
    hotspot = aggregated["hotspot_probability"].to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#fbf8f2")
    ax.hist(hotspot, bins=50, color="#c97b63", edgecolor="#6b2d1f", linewidth=0.4)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("hotspot_probability")
    ax.set_ylabel("Grid-cell count")
    if nearly_constant:
        ax.text(
            0.5,
            0.82,
            "Hotspot probability is nearly constant across the forecast grid.\nThis suggests the 2050 hotspot head is not adding much spatial discrimination yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#6b2d1f",
            bbox={"facecolor": "#f5f1e8", "edgecolor": "#d8c9b8", "boxstyle": "round,pad=0.4"},
        )
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    forecast_csv = Path(args.forecast_csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else forecast_csv.parent / f"{forecast_csv.stem}_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {forecast_csv} in chunks...")
    aggregated = aggregate_forecast_csv(forecast_csv, chunksize=args.chunksize)
    summary = compute_summary(
        aggregated,
        hotspot_quantile=args.hotspot_quantile,
        density_clip_quantile=args.density_clip_quantile,
    )

    aggregated_path = output_dir / "aggregated_forecast_grid.csv"
    aggregated.to_csv(aggregated_path, index=False)
    (output_dir / "forecast_visual_summary.json").write_text(json.dumps(summary, indent=2))

    clip_value = float(summary["predicted_density_clip_value"])
    hotspot_threshold = float(summary["density_hotspot_threshold"])

    save_density_map(
        aggregated,
        output_path=output_dir / "forecast_density_map.png",
        title="FishCast 2050 Forecast Density Map",
        clip_value=clip_value,
    )
    save_hotspot_overlay_map(
        aggregated,
        output_path=output_dir / "forecast_density_hotspot_overlay.png",
        title="FishCast 2050 High-Density Regions",
        clip_value=clip_value,
        hotspot_threshold=hotspot_threshold,
        hotspot_quantile=args.hotspot_quantile,
    )
    save_distribution_plot(
        aggregated,
        output_path=output_dir / "forecast_density_distribution.png",
        title="Distribution of Aggregated 2050 Predicted Density",
    )
    save_hotspot_probability_plot(
        aggregated,
        output_path=output_dir / "forecast_hotspot_probability_distribution.png",
        title="Distribution of Aggregated 2050 Hotspot Probability",
        nearly_constant=bool(summary["hotspot_probability_is_nearly_constant"]),
    )

    print(f"Done. Wrote aggregated grid to {aggregated_path}")
    print(f"Wrote visual summary to {output_dir / 'forecast_visual_summary.json'}")
    print(f"Wrote PNGs to {output_dir}")


if __name__ == "__main__":
    main()
