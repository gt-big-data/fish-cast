from __future__ import annotations

import argparse
from pathlib import Path

from fishcast.visualization import (
    build_forecast_grid,
    load_trained_fishcast,
    predict_grid,
    save_forecast_visualizations,
    save_kernel_visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FishCast forecast and kernel visualizations.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing metrics.json and fishcast_model.pt.")
    parser.add_argument("--data", default="data/krillcast_merged.csv", help="Path to the merged training CSV.")
    parser.add_argument("--output-dir", default="", help="Directory for forecast and kernel outputs. Defaults to <run-dir>/visualizations.")
    parser.add_argument("--baseline-year", type=float, default=None, help="Fractional baseline year used to generate a density change anomaly map.")
    parser.add_argument("--target-year", type=float, default=None, help="Fractional year to visualize. Defaults to the dataset median year.")
    parser.add_argument("--lat-points", type=int, default=120, help="Forecast grid latitude resolution.")
    parser.add_argument("--lon-points", type=int, default=160, help="Forecast grid longitude resolution.")
    parser.add_argument("--year-window", type=float, default=2.0, help="Time window for estimating environmental covariates around target year.")
    parser.add_argument("--neighbors", type=int, default=12, help="KNN neighbors used to estimate grid covariates.")
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"

    model, bundle, _ = load_trained_fishcast(run_dir=run_dir, data_path=args.data, device=args.device)
    forecast_grid = build_forecast_grid(
        bundle=bundle,
        target_year=args.target_year,
        lat_points=args.lat_points,
        lon_points=args.lon_points,
        year_window=args.year_window,
        neighbors=args.neighbors,
    )
    target_year = float(forecast_grid["FRACTIONAL_YEAR"].iloc[0])
    predictions = predict_grid(model=model, bundle=bundle, grid=forecast_grid, device=args.device)
    baseline_predictions = None
    if args.baseline_year is not None:
        baseline_grid = build_forecast_grid(
            bundle=bundle,
            target_year=args.baseline_year,
            lat_points=args.lat_points,
            lon_points=args.lon_points,
            year_window=args.year_window,
            neighbors=args.neighbors,
        )
        baseline_predictions = predict_grid(model=model, bundle=bundle, grid=baseline_grid, device=args.device)
    save_forecast_visualizations(
        predictions=predictions,
        output_dir=output_dir,
        target_year=target_year,
        baseline_predictions=baseline_predictions,
        baseline_year=args.baseline_year,
    )
    save_kernel_visualization(model=model, bundle=bundle, output_dir=output_dir)
    print(f"Saved FishCast visualizations to {output_dir}")


if __name__ == "__main__":
    main()
