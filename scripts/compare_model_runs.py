from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FishCast training runs and flag whether their validation metrics are directly comparable."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories containing metrics.json.",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    return json.loads(metrics_path.read_text())


def describe_validation_unit(metrics: dict[str, Any]) -> str:
    model_name = metrics.get("model_name", "unknown")
    data = metrics.get("data", {})
    if model_name == "autoregressive_sequence":
        return f"sector-year sequences (val={data.get('val_sequences', 'n/a')})"
    if model_name == "convlstm_baseline":
        return f"yearly sector grids (val={data.get('val_sequences', 'n/a')})"
    return f"point observations (val={data.get('val_rows', 'n/a')})"


def compareability_flags(metric_objects: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    model_names = {obj.get("model_name") for obj in metric_objects}
    if ("autoregressive_sequence" in model_names or "convlstm_baseline" in model_names) and len(model_names) > 1:
        flags.append(
            "Sequence/grid baselines are not directly comparable to GP runs because they use temporally windowed aggregates instead of point observations."
        )

    units = {describe_validation_unit(obj) for obj in metric_objects}
    if len(units) > 1:
        flags.append("Validation sample units differ across these runs.")

    val_sizes = set()
    for obj in metric_objects:
        data = obj.get("data", {})
        val_sizes.add(data.get("val_rows", data.get("val_sequences")))
    if len(val_sizes) > 1:
        flags.append("Validation set sizes differ across these runs.")

    return flags


def metric_row(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    data = metrics.get("data", {})
    return {
        "run_dir": str(run_dir),
        "model_name": metrics.get("model_name"),
        "backend": metrics.get("backend"),
        "best_val_rmse": metrics.get("best_validation_rmse"),
        "best_val_auc": metrics.get("best_validation_auc"),
        "checkpoint_epoch": metrics.get("checkpoint_epoch"),
        "checkpoint_val_rmse": metrics.get("checkpoint_validation_rmse"),
        "checkpoint_val_auc": metrics.get("checkpoint_validation_auc"),
        "selected_threshold": metrics.get("selected_validation_threshold"),
        "validation_unit": describe_validation_unit(metrics),
        "seq_len": metrics.get("config", {}).get("seq_len"),
        "horizon_len": metrics.get("config", {}).get("horizon_len"),
        "lag_density_step": metrics.get("config", {}).get("lag_density_step"),
        "lag_mean_window": metrics.get("config", {}).get("lag_mean_window"),
        "lag_hotspot_step": metrics.get("config", {}).get("lag_hotspot_step"),
        "latent_state_dim": metrics.get("config", {}).get("latent_state_dim"),
        "num_inducing_points": data.get("num_inducing_points"),
    }


def print_row(row: dict[str, Any]) -> None:
    print(row["run_dir"])
    print(f"  model:                {row['model_name']}")
    print(f"  backend:              {row['backend']}")
    print(f"  validation unit:      {row['validation_unit']}")
    print(f"  best val RMSE:        {row['best_val_rmse']:.6f}")
    print(f"  best val AUC:         {row['best_val_auc']:.6f}")
    print(f"  checkpoint epoch:     {row['checkpoint_epoch']}")
    print(f"  checkpoint val RMSE:  {row['checkpoint_val_rmse']:.6f}")
    print(f"  checkpoint val AUC:   {row['checkpoint_val_auc']:.6f}")
    print(f"  threshold:            {row['selected_threshold']}")
    if row["model_name"] == "autoregressive_sequence":
        print(f"  seq_len/horizon_len:  {row['seq_len']}/{row['horizon_len']}")
    elif row["model_name"] == "convlstm_baseline":
        print(f"  seq_len/horizon_len:  {row['seq_len']}/{row['horizon_len']}")
    elif row["model_name"] in {"autoregressive_gp", "lag_feature_gp", "latent_state_augmented_gp"}:
        print(
            f"  lag config:           density={row['lag_density_step']}, "
            f"mean_window={row['lag_mean_window']}, hotspot={row['lag_hotspot_step']}"
        )
        if row["model_name"] == "latent_state_augmented_gp":
            print(f"  latent_state_dim:     {row['latent_state_dim']}")
    else:
        print(f"  inducing points:      {row['num_inducing_points']}")


def main() -> None:
    args = parse_args()
    rows = []
    metric_objects = []
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        metrics = load_metrics(run_dir)
        metric_objects.append(metrics)
        rows.append(metric_row(run_dir, metrics))

    print("Comparison")
    for row in rows:
        print_row(row)
        print()

    flags = compareability_flags(metric_objects)
    if flags:
        print("Comparability Warnings")
        for flag in flags:
            print(f"- {flag}")
        print()

    best_rmse = min(rows, key=lambda row: row["best_val_rmse"])
    best_auc = max(rows, key=lambda row: row["best_val_auc"])
    print("Best By Reported Metric")
    print(f"- Lowest best val RMSE: {best_rmse['run_dir']} ({best_rmse['best_val_rmse']:.6f})")
    print(f"- Highest best val AUC: {best_auc['run_dir']} ({best_auc['best_val_auc']:.6f})")


if __name__ == "__main__":
    main()
