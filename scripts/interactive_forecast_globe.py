from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plotly is required for the interactive globe visualization. "
            "Install it in the active environment with `pip install plotly`."
        ) from exc
    return go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive FishCast globe visualization from forecast outputs."
    )
    parser.add_argument(
        "--forecast-csv",
        default="",
        help="Single forecast CSV to visualize as an interactive globe.",
    )
    parser.add_argument(
        "--aggregated-csv",
        default="",
        help="Pre-aggregated forecast grid CSV with one row per LATITUDE/LONGITUDE cell.",
    )
    parser.add_argument(
        "--timelapse-dir",
        default="",
        help="Timelapse directory produced by forecast.py. If frame data exists, an animated globe is created.",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Output HTML path. Defaults beside the selected input.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="Chunk size used if a raw forecast CSV needs aggregation.",
    )
    parser.add_argument(
        "--density-clip-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to clip density color scaling for readability.",
    )
    parser.add_argument(
        "--sample-top-n",
        type=int,
        default=12000,
        help="Maximum number of cells displayed per frame after density-based ranking.",
    )
    parser.add_argument(
        "--title",
        default="FishCast Interactive Globe",
        help="Title shown in the HTML visualization.",
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
    aggregated["predicted_log_density"] = aggregated["predicted_log_density_sum"] / aggregated["duplicate_count"]
    aggregated["hotspot_probability"] = aggregated["hotspot_probability_sum"] / aggregated["duplicate_count"]
    return aggregated.drop(
        columns=["predicted_density_sum", "predicted_log_density_sum", "hotspot_probability_sum"]
    ).sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


def _load_frame_table(path: Path, chunksize: int) -> pd.DataFrame:
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
        if table.duplicated(subset=["LATITUDE", "LONGITUDE"]).any():
            grouped = (
                table.groupby(["LATITUDE", "LONGITUDE"], as_index=False)
                .agg(
                    predicted_density=("predicted_density", "mean"),
                    predicted_log_density=("predicted_log_density", "mean"),
                    hotspot_probability=("hotspot_probability", "mean"),
                )
            )
            table = grouped
    elif path.suffix == ".csv":
        preview = pd.read_csv(path, nrows=1000)
        if preview.duplicated(subset=["LATITUDE", "LONGITUDE"]).any():
            return aggregate_forecast_csv(path, chunksize=chunksize)
        table = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported frame-data file: {path}")
    return table.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


def _select_globe_mode(table: pd.DataFrame) -> str:
    return "density_change" if "density_change" in table.columns else "predicted_density"


def _select_density_change_columns(table: pd.DataFrame) -> tuple[str, str]:
    density_columns = sorted(column for column in table.columns if column.startswith("predicted_density_"))
    if len(density_columns) < 2:
        raise ValueError("Density change globe inputs require baseline and target predicted density columns.")
    return density_columns[0], density_columns[-1]


def _resolve_single_source(args: argparse.Namespace) -> tuple[str, Dict[str, pd.DataFrame], Path]:
    if args.timelapse_dir:
        timelapse_dir = Path(args.timelapse_dir)
        manifest = json.loads((timelapse_dir / "timelapse_manifest.json").read_text())
        frame_data_dir = timelapse_dir / "frame_data"
        if not frame_data_dir.exists():
            raise FileNotFoundError(
                f"{frame_data_dir} does not exist. Re-run forecast.py with "
                "`--write-frame-data --output-format parquet` to build an animated interactive globe."
            )

        frame_tables: Dict[str, pd.DataFrame] = {}
        for frame in manifest["frames"]:
            year = str(int(float(frame["forecast_year"])))
            parquet_path = frame_data_dir / f"forecast_{year}.parquet"
            csv_path = frame_data_dir / f"forecast_{year}.csv"
            if parquet_path.exists():
                frame_tables[year] = _load_frame_table(parquet_path, chunksize=args.chunksize)
            elif csv_path.exists():
                frame_tables[year] = _load_frame_table(csv_path, chunksize=args.chunksize)
        if not frame_tables:
            raise FileNotFoundError(
                f"No per-frame data files were found in {frame_data_dir}. "
                "Re-run forecast.py with `--write-frame-data`."
            )
        output_html = (
            Path(args.output_html)
            if args.output_html
            else timelapse_dir / "interactive_forecast_globe.html"
        )
        return args.title, frame_tables, output_html

    if args.aggregated_csv:
        aggregated_path = Path(args.aggregated_csv)
        frame_tables = {aggregated_path.stem: _load_frame_table(aggregated_path, chunksize=args.chunksize)}
        output_html = (
            Path(args.output_html)
            if args.output_html
            else aggregated_path.with_name(f"{aggregated_path.stem}_interactive_globe.html")
        )
        return args.title, frame_tables, output_html

    if args.forecast_csv:
        forecast_path = Path(args.forecast_csv)
        frame_tables = {forecast_path.stem: aggregate_forecast_csv(forecast_path, chunksize=args.chunksize)}
        output_html = (
            Path(args.output_html)
            if args.output_html
            else forecast_path.with_name(f"{forecast_path.stem}_interactive_globe.html")
        )
        return args.title, frame_tables, output_html

    raise ValueError("Provide one of --timelapse-dir, --aggregated-csv, or --forecast-csv.")


def _load_support_points(path: Path) -> pd.DataFrame:
    support = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"]).dropna().drop_duplicates()
    return support.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


def _filter_to_supported_cells(
    frame_tables: Dict[str, pd.DataFrame],
    support_points: pd.DataFrame,
    max_distance_deg: float,
) -> Dict[str, pd.DataFrame]:
    if max_distance_deg <= 0.0 or support_points.empty:
        return frame_tables

    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(support_points[["LATITUDE", "LONGITUDE"]])

    filtered_tables: Dict[str, pd.DataFrame] = {}
    for name, table in frame_tables.items():
        distances, _ = neighbors.kneighbors(table[["LATITUDE", "LONGITUDE"]])
        keep_mask = distances[:, 0] <= max_distance_deg
        filtered_tables[name] = table.loc[keep_mask].copy().reset_index(drop=True)
    return filtered_tables


def _projected_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    # The CCAMLR file is tagged as EPSG:6932 and behaves like a south-polar
    # stereographic Antarctic projection for these MPA polygons.
    earth_radius_m = 6378137.0
    rho = math.hypot(x, y)
    longitude = math.degrees(math.atan2(x, y))
    latitude = math.degrees((2.0 * math.atan2(rho, 2.0 * earth_radius_m)) - (math.pi / 2.0))
    return longitude, latitude


def _load_mpa_features(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    features = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            continue
        rings = []
        for ring in geometry.get("coordinates", []):
            lon_lat_ring = [_projected_to_lon_lat(x, y) for x, y in ring]
            rings.append(lon_lat_ring)
        if not rings:
            continue
        properties = feature.get("properties", {})
        features.append(
            {
                "name": properties.get("GAR_Name") or properties.get("GAR_Short_Label") or feature.get("id", "CCAMLR MPA"),
                "short_label": properties.get("GAR_Short_Label", ""),
                "reference": properties.get("GAR_Reference", ""),
                "description": properties.get("GAR_Description", ""),
                "rings": rings,
            }
        )
    return features


def _mpa_traces(go, mpa_features: Sequence[dict]) -> List[object]:
    traces = []
    for index, feature in enumerate(mpa_features):
        outer_ring = feature["rings"][0]
        lons = [lon for lon, _ in outer_ring]
        lats = [lat for _, lat in outer_ring]
        hover_text = (
            f"{feature['name']}"
            + (f"<br>Label: {feature['short_label']}" if feature["short_label"] else "")
            + (f"<br>Reference: {feature['reference']}" if feature["reference"] else "")
            + (f"<br>{feature['description']}" if feature["description"] else "")
        )
        traces.append(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line={"color": "#6f6f6f", "width": 1.8},
                name="CCAMLR MPAs" if index == 0 else feature["name"],
                legendgroup="camlr_mpas",
                showlegend=index == 0,
                visible=True,
                text=[hover_text] * len(lons),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    return traces


def _prepare_frame(
    table: pd.DataFrame,
    display_clip_value: float,
    sample_top_n: int,
    mode: str,
) -> pd.DataFrame:
    working = table.copy()
    if mode == "density_change":
        working["density_change_clipped"] = np.clip(
            working["density_change"],
            -display_clip_value,
            display_clip_value,
        )
        if display_clip_value <= 0.0:
            working["density_change_display"] = 0.0
        else:
            working["density_change_display"] = working["density_change_clipped"] / display_clip_value
        working["marker_size"] = np.clip(np.abs(working["density_change_display"]) * 14.0 + 3.0, 3.0, 18.0)
        working["sampling_rank"] = np.abs(working["density_change"])
    else:
        working["density_clipped"] = np.clip(
            working["predicted_density"],
            0.0,
            display_clip_value,
        )
        working["density_display"] = np.log10(working["density_clipped"] + 1.0)
        working["marker_size"] = np.clip((working["density_display"] + 0.3) * 5.0, 3.0, 18.0)
        working["sampling_rank"] = working["predicted_density"]
    working = working.sort_values("sampling_rank", ascending=False).reset_index(drop=True)
    if len(working) > sample_top_n:
        indices = np.linspace(0, len(working) - 1, num=sample_top_n, dtype=int)
        working = working.iloc[np.unique(indices)].copy()
    return working


def _trace_for_frame(go, frame_name: str, table: pd.DataFrame, cmin: float, cmax: float, mode: str):
    if mode == "density_change":
        baseline_column, target_column = _select_density_change_columns(table)
        marker = {
            "size": table["marker_size"],
            "color": table["density_change_display"],
            "cmin": cmin,
            "cmax": cmax,
            "colorscale": [
                [0.0, "#b2182b"],
                [0.15, "#d6604d"],
                [0.35, "#f4a582"],
                [0.5, "#ffffff"],
                [0.65, "#92c5de"],
                [0.85, "#4393c3"],
                [1.0, "#2166ac"],
            ],
            "opacity": 0.9,
            "line": {"width": 0},
            "colorbar": {
                "title": "Density change",
                "x": 0.98,
                "y": 0.5,
                "len": 0.75,
                "bgcolor": "rgba(255,255,255,0.7)",
            },
        }
        text = [
            f"Frame: {frame_name}<br>Lat: {lat:.1f}<br>Lon: {lon:.1f}<br>"
            f"{baseline_column.replace('_', ' ').title()}: {historical:,.3f}<br>"
            f"{target_column.replace('_', ' ').title()}: {future:,.3f}<br>"
            f"Density change: {change:+,.3f}"
            for lat, lon, historical, future, change in zip(
                table["LATITUDE"],
                table["LONGITUDE"],
                table[baseline_column],
                table[target_column],
                table["density_change"],
            )
        ]
    else:
        marker = {
            "size": table["marker_size"],
            "color": table["density_display"],
            "cmin": cmin,
            "cmax": cmax,
            "colorscale": [
                [0.0, "#0b1f33"],
                [0.15, "#145374"],
                [0.35, "#2d8f9d"],
                [0.55, "#7bc8a4"],
                [0.75, "#f2c14e"],
                [0.9, "#f78154"],
                [1.0, "#c5283d"],
            ],
            "opacity": 0.9,
            "line": {"width": 0},
            "colorbar": {
                "title": "log10(1 + density)",
                "x": 0.98,
                "y": 0.5,
                "len": 0.75,
                "bgcolor": "rgba(255,255,255,0.7)",
            },
        }
        text = [
            f"Year: {frame_name}<br>Lat: {lat:.1f}<br>Lon: {lon:.1f}<br>"
            f"Predicted density: {density:,.3f}<br>"
            f"Hotspot probability: {hotspot:.4f}"
            for lat, lon, density, hotspot in zip(
                table["LATITUDE"],
                table["LONGITUDE"],
                table["predicted_density"],
                table["hotspot_probability"],
            )
        ]
    return go.Scattergeo(
        lon=table["LONGITUDE"],
        lat=table["LATITUDE"],
        mode="markers",
        marker=marker,
        text=text,
        hovertemplate="%{text}<extra></extra>",
        name=frame_name,
    )


def build_interactive_globe(
    frame_tables: Dict[str, pd.DataFrame],
    output_html: Path,
    title: str,
    density_clip_quantile: float,
    sample_top_n: int,
    mpa_features: Sequence[dict] | None = None,
) -> None:
    go = _require_plotly()
    ordered_names = sorted(frame_tables.keys(), key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value)
    mode = _select_globe_mode(frame_tables[ordered_names[0]])
    if mode == "density_change":
        global_clip_value = float(
            np.quantile(
                np.concatenate([np.abs(frame_tables[name]["density_change"].to_numpy()) for name in ordered_names]),
                density_clip_quantile,
            )
        )
    else:
        global_clip_value = float(
            np.quantile(
                np.concatenate([frame_tables[name]["predicted_density"].to_numpy() for name in ordered_names]),
                density_clip_quantile,
            )
        )
    prepared = {
        name: _prepare_frame(frame_tables[name], display_clip_value=global_clip_value, sample_top_n=sample_top_n, mode=mode)
        for name in ordered_names
    }
    if mode == "density_change":
        global_cmin = -1.0
        global_cmax = 1.0
    else:
        global_cmin = 0.0
        global_cmax = float(max(prepared[name]["density_display"].max() for name in ordered_names))

    first_name = ordered_names[0]
    first_trace = _trace_for_frame(go, first_name, prepared[first_name], cmin=global_cmin, cmax=global_cmax, mode=mode)
    mpa_trace_list = _mpa_traces(go, mpa_features or [])
    frames = [
        go.Frame(
            name=name,
            data=[_trace_for_frame(go, name, prepared[name], cmin=global_cmin, cmax=global_cmax, mode=mode)],
            traces=[0],
        )
        for name in ordered_names
    ]

    slider_steps = [
        {
            "label": name,
            "method": "animate",
            "args": [[name], {"mode": "immediate", "frame": {"duration": 450, "redraw": True}, "transition": {"duration": 250}}],
        }
        for name in ordered_names
    ]

    updatemenus = [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.1,
            "y": 0.04,
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "transition": {"duration": 250},
                            "fromcurrent": True,
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                },
            ],
        }
    ]
    if mpa_trace_list:
        updatemenus.append(
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.60,
                "y": 0.04,
                "showactive": True,
                "buttons": [
                    {
                        "label": "Show MPAs",
                        "method": "restyle",
                        "args": [{"visible": True}, list(range(1, len(mpa_trace_list) + 1))],
                    },
                    {
                        "label": "Hide MPAs",
                        "method": "restyle",
                        "args": [{"visible": False}, list(range(1, len(mpa_trace_list) + 1))],
                    },
                ],
            }
        )

    fig = go.Figure(
        data=[first_trace, *mpa_trace_list],
        frames=frames,
        layout=go.Layout(
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
            updatemenus=updatemenus,
            sliders=[
                {
                    "active": 0,
                    "x": 0.12,
                    "y": 0.03,
                    "len": 0.75,
                    "currentvalue": {"prefix": "Frame: ", "font": {"size": 15}},
                    "pad": {"b": 10, "t": 40},
                    "steps": slider_steps,
                }
            ],
            annotations=[
                {
                    "text": "Drag to rotate the globe. Scroll to zoom. Slider or Play animates forecast years. Use the MPA buttons to show or hide CCAMLR protected areas.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.0,
                    "showarrow": False,
                    "font": {"size": 12, "color": "#4a4a4a"},
                }
            ],
        ),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    title, frame_tables, output_html = _resolve_single_source(args)
    support_points = _load_support_points(Path(args.support_csv)) if args.support_csv else pd.DataFrame()
    frame_tables = _filter_to_supported_cells(
        frame_tables=frame_tables,
        support_points=support_points,
        max_distance_deg=args.max_distance_deg,
    )
    mpa_features = _load_mpa_features(Path(args.mpa_json)) if args.mpa_json else []
    build_interactive_globe(
        frame_tables=frame_tables,
        output_html=output_html,
        title=title,
        density_clip_quantile=args.density_clip_quantile,
        sample_top_n=args.sample_top_n,
        mpa_features=mpa_features,
    )
    print(f"Wrote interactive globe to {output_html}")


if __name__ == "__main__":
    main()
