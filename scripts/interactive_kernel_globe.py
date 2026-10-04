from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plotly is required for the interactive kernel globe. "
            "Install it in the active environment with `pip install plotly`."
        ) from exc
    return go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive globe visualization of FishCast kernel ellipses."
    )
    parser.add_argument(
        "--kernel-csv",
        required=True,
        help="Path to kernel_covariance_matrices.csv produced by analyze_kernel_geometry.py.",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Output HTML path. Defaults beside the CSV.",
    )
    parser.add_argument(
        "--sample-top-n",
        type=int,
        default=220,
        help="Maximum number of ellipses to render, ranked by anisotropy.",
    )
    parser.add_argument(
        "--max-major-radius-deg",
        type=float,
        default=10.0,
        help="Displayed major-axis radius cap in degrees for readability on the globe.",
    )
    parser.add_argument(
        "--min-major-radius-deg",
        type=float,
        default=2.0,
        help="Displayed minimum major-axis radius in degrees.",
    )
    parser.add_argument(
        "--perimeter-points",
        type=int,
        default=48,
        help="Number of points used to draw each ellipse perimeter.",
    )
    parser.add_argument(
        "--color-by",
        choices=["anisotropy_ratio", "east_west_alignment", "ellipse_angle_degrees"],
        default="east_west_alignment",
        help="Which kernel diagnostic drives the marker color.",
    )
    parser.add_argument(
        "--title",
        default="FishCast Kernel Geometry Globe",
        help="Title shown in the HTML visualization.",
    )
    return parser.parse_args()


def _wrap_longitudes(values: np.ndarray) -> np.ndarray:
    wrapped = ((values + 180.0) % 360.0) - 180.0
    return wrapped


def _ellipse_display_radii(
    table: pd.DataFrame,
    min_major_radius_deg: float,
    max_major_radius_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    major = table["major_std_degrees"].to_numpy()
    minor = table["minor_std_degrees"].to_numpy()
    major_scaled = np.clip(major, np.quantile(major, 0.15), np.quantile(major, 0.95))
    major_display = min_major_radius_deg + (
        (major_scaled - major_scaled.min()) / max(major_scaled.max() - major_scaled.min(), 1e-9)
    ) * (max_major_radius_deg - min_major_radius_deg)
    ratio = np.clip(minor / np.maximum(major, 1e-9), 0.03, 1.0)
    minor_display = major_display * ratio
    return major_display, minor_display


def _ellipse_perimeter(
    center_lat: float,
    center_lon: float,
    major_radius: float,
    minor_radius: float,
    angle_deg: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    x = major_radius * np.cos(theta)
    y = minor_radius * np.sin(theta)
    angle = np.radians(angle_deg)
    rot_x = x * np.cos(angle) - y * np.sin(angle)
    rot_y = x * np.sin(angle) + y * np.cos(angle)

    lat = center_lat + rot_y
    lon = center_lon + rot_x / np.maximum(np.cos(np.radians(center_lat)), 0.15)
    lon = _wrap_longitudes(lon)
    return lat, lon


def build_kernel_globe(
    table: pd.DataFrame,
    output_html: Path,
    sample_top_n: int,
    min_major_radius_deg: float,
    max_major_radius_deg: float,
    perimeter_points: int,
    color_by: str,
    title: str,
) -> None:
    go = _require_plotly()
    working = table.sort_values("anisotropy_ratio", ascending=False).head(sample_top_n).copy()
    major_display, minor_display = _ellipse_display_radii(
        working,
        min_major_radius_deg=min_major_radius_deg,
        max_major_radius_deg=max_major_radius_deg,
    )
    working["major_display_radius_deg"] = major_display
    working["minor_display_radius_deg"] = minor_display
    working["wrapped_longitude"] = _wrap_longitudes(working["LONGITUDE"].to_numpy())

    ellipse_lats: list[float | None] = []
    ellipse_lons: list[float | None] = []
    for row in working.itertuples(index=False):
        lat, lon = _ellipse_perimeter(
            center_lat=float(row.LATITUDE),
            center_lon=float(row.wrapped_longitude),
            major_radius=float(row.major_display_radius_deg),
            minor_radius=float(row.minor_display_radius_deg),
            angle_deg=float(row.ellipse_angle_degrees),
            num_points=perimeter_points,
        )
        ellipse_lats.extend(lat.tolist() + [None])
        ellipse_lons.extend(lon.tolist() + [None])

    colorbar_title = {
        "anisotropy_ratio": "Anisotropy",
        "east_west_alignment": "E-W alignment",
        "ellipse_angle_degrees": "Angle (deg)",
    }[color_by]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=ellipse_lons,
            lat=ellipse_lats,
            mode="lines",
            line={"width": 1.0, "color": "rgba(255, 205, 96, 0.45)"},
            hoverinfo="skip",
            name="Kernel ellipses",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=working["wrapped_longitude"],
            lat=working["LATITUDE"],
            mode="markers",
            marker={
                "size": np.clip(working["major_display_radius_deg"] * 1.5, 5.0, 18.0),
                "color": working[color_by],
                "colorscale": [
                    [0.0, "#0b1f33"],
                    [0.2, "#145374"],
                    [0.45, "#2d8f9d"],
                    [0.65, "#7bc8a4"],
                    [0.82, "#f2c14e"],
                    [1.0, "#f78154"],
                ],
                "opacity": 0.95,
                "line": {"width": 0.5, "color": "rgba(245, 239, 228, 0.7)"},
                "colorbar": {
                    "title": colorbar_title,
                    "x": 0.98,
                    "y": 0.5,
                    "len": 0.75,
                    "bgcolor": "rgba(255,255,255,0.7)",
                },
            },
            text=[
                "Lat: {:.1f}<br>Lon: {:.1f}<br>"
                "Anisotropy: {:.2f}<br>"
                "E-W alignment: {:.3f}<br>"
                "Kernel angle: {:.1f} deg<br>"
                "Raw major std: {:.2f} deg<br>"
                "Raw minor std: {:.2f} deg".format(
                    lat,
                    lon,
                    anisotropy,
                    alignment,
                    angle,
                    major_std,
                    minor_std,
                )
                for lat, lon, anisotropy, alignment, angle, major_std, minor_std in zip(
                    working["LATITUDE"],
                    working["wrapped_longitude"],
                    working["anisotropy_ratio"],
                    working["east_west_alignment"],
                    working["ellipse_angle_degrees"],
                    working["major_std_degrees"],
                    working["minor_std_degrees"],
                )
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Kernel centers",
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24},
        },
        paper_bgcolor="#f4efe4",
        plot_bgcolor="#f4efe4",
        margin={"l": 10, "r": 10, "t": 70, "b": 20},
        geo={
            "projection": {"type": "orthographic", "rotation": {"lon": 20, "lat": -62}},
            "showland": True,
            "landcolor": "#f7f4ea",
            "showocean": True,
            "oceancolor": "#0b1f33",
            "showlakes": True,
            "lakecolor": "#0b1f33",
            "showcoastlines": True,
            "coastlinecolor": "#d9d2c3",
            "coastlinewidth": 0.6,
            "bgcolor": "#f4efe4",
            "showframe": False,
            "lonaxis": {"showgrid": False},
            "lataxis": {"showgrid": False},
        },
        annotations=[
            {
                "text": "Ellipses are display-scaled for readability. Orientation and anisotropy come from the learned kernel; size is clipped visually.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.0,
                "showarrow": False,
                "font": {"size": 12, "color": "#4a4a4a"},
            }
        ],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.02,
            "xanchor": "left",
            "x": 0.02,
            "bgcolor": "rgba(255,255,255,0.55)",
        },
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    kernel_csv = Path(args.kernel_csv)
    table = pd.read_csv(kernel_csv)
    output_html = (
        Path(args.output_html)
        if args.output_html
        else kernel_csv.with_name(f"{kernel_csv.stem}_interactive_globe.html")
    )
    build_kernel_globe(
        table=table,
        output_html=output_html,
        sample_top_n=args.sample_top_n,
        min_major_radius_deg=args.min_major_radius_deg,
        max_major_radius_deg=args.max_major_radius_deg,
        perimeter_points=args.perimeter_points,
        color_by=args.color_by,
        title=args.title,
    )
    print(f"Wrote interactive kernel globe to {output_html}")


if __name__ == "__main__":
    main()
