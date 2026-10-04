from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fishcast.visualization import load_trained_fishcast, spatial_kernel_covariances_latlon


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is required for kernel geometry diagnostics. Install it in the active environment first."
        ) from exc
    return plt, Ellipse


def _normalize_column(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (values - stats["mean"]) / stats["std"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FishCast spatial covariance matrices and visualize learned kernel ellipses."
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory with metrics.json and fishcast_model.pt.")
    parser.add_argument("--data", default="data/krillcast_merged.csv", help="Path to the merged training CSV.")
    parser.add_argument("--output-dir", default="", help="Directory for kernel outputs. Defaults to <run-dir>/kernel_analysis.")
    parser.add_argument("--lat-points", type=int, default=20, help="Latitude resolution for kernel sampling.")
    parser.add_argument("--lon-points", type=int, default=28, help="Longitude resolution for kernel sampling.")
    parser.add_argument("--ellipse-scale", type=float, default=2.5, help="Scale multiplier for plotted ellipse radii.")
    parser.add_argument("--device", default="cpu", help="Torch device for model loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "kernel_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, bundle, metrics = load_trained_fishcast(run_dir=run_dir, data_path=args.data, device=args.device)
    frame = bundle.dataframe
    latitudes = np.linspace(frame["LATITUDE"].min(), frame["LATITUDE"].max(), args.lat_points)
    longitudes = np.linspace(frame["LONGITUDE"].min(), frame["LONGITUDE"].max(), args.lon_points)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    covariance_matrices = spatial_kernel_covariances_latlon(model, bundle, lat_grid.ravel(), lon_grid.ravel())

    rows = []
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
        major = float(np.sqrt(max(eigenvalues[0], 1e-12)))
        minor = float(np.sqrt(max(eigenvalues[1], 1e-12)))
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
        angle_wrapped = ((angle + 90.0) % 180.0) - 90.0
        east_west_alignment = float(np.cos(np.radians(angle_wrapped)) ** 2)
        anisotropy_ratio = float(major / max(minor, 1e-12))

        rows.append(
            {
                "LATITUDE": float(latitude),
                "LONGITUDE": float(longitude),
                "sigma_lat_lat": float(sigma_raw[0, 0]),
                "sigma_lat_lon": float(sigma_raw[0, 1]),
                "sigma_lon_lat": float(sigma_raw[1, 0]),
                "sigma_lon_lon": float(sigma_raw[1, 1]),
                "major_std_degrees": major,
                "minor_std_degrees": minor,
                "anisotropy_ratio": anisotropy_ratio,
                "ellipse_angle_degrees": angle_wrapped,
                "east_west_alignment": east_west_alignment,
                "ellipse_width_degrees": args.ellipse_scale * 2.0 * major,
                "ellipse_height_degrees": args.ellipse_scale * 2.0 * minor,
            }
        )

    ellipse_df = pd.DataFrame(rows)
    ellipse_df.to_csv(output_dir / "kernel_covariance_matrices.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "model_name": metrics.get("model_name"),
        "mean_anisotropy_ratio": float(ellipse_df["anisotropy_ratio"].mean()),
        "median_anisotropy_ratio": float(ellipse_df["anisotropy_ratio"].median()),
        "max_anisotropy_ratio": float(ellipse_df["anisotropy_ratio"].max()),
        "mean_east_west_alignment": float(ellipse_df["east_west_alignment"].mean()),
        "median_east_west_alignment": float(ellipse_df["east_west_alignment"].median()),
        "fraction_strongly_east_west": float((ellipse_df["east_west_alignment"] >= 0.75).mean()),
        "fraction_highly_anisotropic": float((ellipse_df["anisotropy_ratio"] >= 2.0).mean()),
        "lat_points": args.lat_points,
        "lon_points": args.lon_points,
    }
    (output_dir / "kernel_geometry_summary.json").write_text(json.dumps(summary, indent=2))

    plt, Ellipse = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#fbf8f2")
    ax.scatter(frame["LONGITUDE"], frame["LATITUDE"], s=3, alpha=0.06, color="#2f3e46")
    for row in ellipse_df.itertuples(index=False):
        ellipse = Ellipse(
            xy=(row.LONGITUDE, row.LATITUDE),
            width=row.ellipse_width_degrees,
            height=row.ellipse_height_degrees,
            angle=row.ellipse_angle_degrees,
            fill=False,
            linewidth=0.9,
            alpha=0.75,
            edgecolor="#d94841",
        )
        ax.add_patch(ellipse)
    ax.set_title("FishCast Learned Kernel Ellipses", fontsize=16, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(frame["LONGITUDE"].min(), frame["LONGITUDE"].max())
    ax.set_ylim(frame["LATITUDE"].min(), frame["LATITUDE"].max())
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_ellipses_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#fbf8f2")
    scatter = ax.scatter(
        ellipse_df["LONGITUDE"],
        ellipse_df["LATITUDE"],
        c=ellipse_df["east_west_alignment"],
        s=np.clip(ellipse_df["anisotropy_ratio"] * 20.0, 20.0, 120.0),
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("East-west alignment score")
    ax.set_title("Kernel Alignment Toward Antarctic Circumpolar Current Geometry", fontsize=16, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_alignment_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("#f5f1e8")
    for ax in axes:
        ax.set_facecolor("#fbf8f2")
    axes[0].hist(ellipse_df["anisotropy_ratio"], bins=30, color="#2d8f9d", edgecolor="#0b1f33", linewidth=0.4)
    axes[0].set_title("Anisotropy Ratio")
    axes[0].set_xlabel("major / minor")
    axes[0].set_ylabel("Sample count")
    axes[1].hist(ellipse_df["east_west_alignment"], bins=30, color="#f4a261", edgecolor="#7f5539", linewidth=0.4)
    axes[1].set_title("East-West Alignment")
    axes[1].set_xlabel("cos(angle)^2")
    axes[1].set_ylabel("Sample count")
    fig.suptitle("Kernel Geometry Diagnostics", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_diagnostic_histograms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote covariance matrices to {output_dir / 'kernel_covariance_matrices.csv'}")
    print(f"Wrote summary to {output_dir / 'kernel_geometry_summary.json'}")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
