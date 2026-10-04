from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from fishcast import TrainingConfig, load_autoregressive_dataset, load_convlstm_dataset, load_krill_dataset, train_model


def _parse_float_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FishCast krill forecasting model.")
    parser.add_argument(
        "--model-type",
        choices=[
            "climate_conditioned_gp",
            "autoregressive_gp",
            "lag_feature_gp",
            "latent_state_augmented_gp",
            "autoregressive_sequence",
            "convlstm_baseline",
        ],
        default="climate_conditioned_gp",
        help="Which FishCast model family member to train.",
    )
    parser.add_argument("--data", default="data/krillcast_merged.csv", help="Path to the merged krill dataset.")
    parser.add_argument("--output-dir", default="artifacts/fishcast_run", help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--classification-weight", type=float, default=1.0, help="Relative hotspot loss weight.")
    parser.add_argument("--classification-weights", default="", help="Comma-separated classification weights for an experiment sweep.")
    parser.add_argument("--num-inducing-points", type=int, default=500, help="Number of K-means initialized inducing points.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size for the deep spatial kernel.")
    parser.add_argument("--validation-fraction", type=float, default=0.15, help="Fraction of latest observations reserved for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of latest observations held out as a test set (GP models). Never used for tuning.")
    parser.add_argument("--spatial-coords", choices=["polar", "latlon"], default="polar", help="Spatial coordinates for the GP kernel. 'polar' avoids the +/-180 longitude seam.")
    parser.add_argument("--hotspot-quantile", type=float, default=0.9, help="Sector-level quantile used to define hotspots.")
    parser.add_argument("--lat-sector-step", type=float, default=5.0, help="Latitude bin size for hotspot sectors.")
    parser.add_argument("--lon-sector-step", type=float, default=5.0, help="Longitude bin size for hotspot sectors.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="ReduceLROnPlateau patience.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--scheduler-monitor", choices=["val_rmse", "val_auc"], default="val_rmse", help="Validation metric used by the scheduler.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5, help="Lower bound for the learning rate scheduler.")
    parser.add_argument("--threshold-steps", type=int, default=41, help="Number of validation thresholds tested between 0.05 and 0.95.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for GPU acceleration.")
    parser.add_argument("--seq-len", type=int, default=5, help="Context length for autoregressive sequence training.")
    parser.add_argument("--horizon-len", type=int, default=3, help="Forecast horizon length for autoregressive sequence training.")
    parser.add_argument("--autoregressive-hidden-dim", type=int, default=64, help="Hidden size for the autoregressive GRU forecaster.")
    parser.add_argument("--lag-density-step", type=int, default=1, help="Lag step used for lagged krill density in lag_feature_gp.")
    parser.add_argument("--lag-mean-window", type=int, default=3, help="Rolling lag window used for mean krill density in lag_feature_gp.")
    parser.add_argument("--lag-hotspot-step", type=int, default=1, help="Lag step used for lagged hotspot state in lag_feature_gp.")
    parser.add_argument("--latent-state-dim", type=int, default=8, help="Latent ecological state width for latent_state_augmented_gp.")
    return parser.parse_args()


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    canonical_model_type = "autoregressive_gp" if args.model_type == "lag_feature_gp" else args.model_type
    if canonical_model_type == "autoregressive_sequence":
        bundle = load_autoregressive_dataset(
            csv_path=args.data,
            seq_len=args.seq_len,
            horizon_len=args.horizon_len,
            hotspot_quantile=args.hotspot_quantile,
            validation_fraction=args.validation_fraction,
            lat_sector_step=args.lat_sector_step,
            lon_sector_step=args.lon_sector_step,
        )
    elif canonical_model_type == "convlstm_baseline":
        bundle = load_convlstm_dataset(
            csv_path=args.data,
            seq_len=args.seq_len,
            horizon_len=args.horizon_len,
            hotspot_quantile=args.hotspot_quantile,
            validation_fraction=args.validation_fraction,
            lat_sector_step=args.lat_sector_step,
            lon_sector_step=args.lon_sector_step,
        )
    else:
        bundle = load_krill_dataset(
            csv_path=args.data,
            num_inducing_points=args.num_inducing_points,
            hotspot_quantile=args.hotspot_quantile,
            validation_fraction=args.validation_fraction,
            lat_sector_step=args.lat_sector_step,
            lon_sector_step=args.lon_sector_step,
            include_lag_features=canonical_model_type in {"autoregressive_gp", "latent_state_augmented_gp"},
            lag_density_step=args.lag_density_step,
            lag_mean_window=args.lag_mean_window,
            lag_hotspot_step=args.lag_hotspot_step,
            test_fraction=args.test_fraction,
            spatial_coords=args.spatial_coords,
        )
    classification_weights = (
        _parse_float_list(args.classification_weights)
        if args.classification_weights.strip()
        else [args.classification_weight]
    )

    if len(classification_weights) == 1:
        summary = train_model(
            bundle=bundle,
            config=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                classification_weight=classification_weights[0],
                model_name=canonical_model_type,
                hidden_dim=args.hidden_dim,
                device=args.device,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                scheduler_monitor=args.scheduler_monitor,
                scheduler_min_lr=args.scheduler_min_lr,
                threshold_steps=args.threshold_steps,
                compile_model=args.compile,
                seq_len=args.seq_len,
                horizon_len=args.horizon_len,
                autoregressive_hidden_dim=args.autoregressive_hidden_dim,
                lag_density_step=args.lag_density_step,
                lag_mean_window=args.lag_mean_window,
                lag_hotspot_step=args.lag_hotspot_step,
                latent_state_dim=args.latent_state_dim,
            ),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(summary, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results = []
    for weight in classification_weights:
        run_dir = output_dir / f"classification_weight_{str(weight).replace('.', '_')}"
        summary = train_model(
            bundle=bundle,
            config=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                classification_weight=weight,
                model_name=canonical_model_type,
                hidden_dim=args.hidden_dim,
                device=args.device,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                scheduler_monitor=args.scheduler_monitor,
                scheduler_min_lr=args.scheduler_min_lr,
                threshold_steps=args.threshold_steps,
                compile_model=args.compile,
                seq_len=args.seq_len,
                horizon_len=args.horizon_len,
                autoregressive_hidden_dim=args.autoregressive_hidden_dim,
                lag_density_step=args.lag_density_step,
                lag_mean_window=args.lag_mean_window,
                lag_hotspot_step=args.lag_hotspot_step,
                latent_state_dim=args.latent_state_dim,
            ),
            output_dir=run_dir,
        )
        sweep_results.append(
            {
                "classification_weight": weight,
                "output_dir": str(run_dir),
                "best_validation_rmse": summary["best_validation_rmse"],
                "best_validation_auc": summary["best_validation_auc"],
                "selected_validation_threshold": summary["selected_validation_threshold"],
            }
        )

    best_by_auc = max(sweep_results, key=lambda row: row["best_validation_auc"])
    best_by_rmse = min(sweep_results, key=lambda row: row["best_validation_rmse"])
    sweep_summary = {
        "runs": sweep_results,
        "best_by_auc": best_by_auc,
        "best_by_rmse": best_by_rmse,
    }
    (output_dir / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2))
    print(json.dumps(sweep_summary, indent=2))


if __name__ == "__main__":
    main()
