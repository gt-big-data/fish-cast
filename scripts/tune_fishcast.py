import argparse
import json
import functools
from pathlib import Path

import optuna
import torch

from fishcast import TrainingConfig, load_krill_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for FishCast.")
    parser.add_argument("--data", default="data/krillcast_merged.csv", help="Path to the merged krill dataset.")
    parser.add_argument("--output-dir", default="artifacts/FishCast_Optuna_Sweep", help="Directory for trial checkpoints.")
    parser.add_argument("--study-name", default="fishcast_h200_sweep", help="Optuna study name.")
    parser.add_argument("--storage", default="", help="Optuna storage URL. Defaults to a sqlite file inside output-dir.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials to run.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per trial.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size per trial.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device for trials.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() during trials.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate validation metrics every N epochs during tuning.")
    parser.add_argument("--tune-threshold-interval", type=int, default=50, help="Retune hotspot threshold every N epochs during tuning.")
    parser.add_argument("--dataloader-workers", type=int, default=4, help="Number of DataLoader workers per trial.")
    return parser.parse_args()


@functools.lru_cache(maxsize=4)
def get_cached_dataset(csv_path: str, num_inducing_points: int):
    print(f"\n[Cache Miss] Loading CSV and running KMeans for {num_inducing_points} points...", flush=True)
    return load_krill_dataset(csv_path=csv_path, num_inducing_points=num_inducing_points)


def main() -> None:
    # Enable TF32 for massive speedups on the NVIDIA H200
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_url = args.storage or f"sqlite:///{(output_dir / 'optuna_study.db').resolve()}"

    def objective(trial: optuna.Trial) -> float:
        print(f"\n--- Starting Trial #{trial.number} ---", flush=True)

        # 1. Optuna suggests hyperparameters for this specific trial
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        num_inducing_points = trial.suggest_categorical("num_inducing_points", [100, 250, 500, 1000])
        classification_weight = trial.suggest_float("classification_weight", 1.0, 20.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # 2. Prepare Data (Re-runs K-means quickly for the suggested number of inducing points)
        bundle = get_cached_dataset(args.data, num_inducing_points)

        # 3. Configure Training (with torch.compile enabled for the H200!)
        trial_dir = output_dir / f"trial_{trial.number}"
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            classification_weight=classification_weight,
            hidden_dim=hidden_dim,
            device=args.device,
            scheduler_monitor="val_auc",
            checkpoint_monitor="val_auc",
            compile_model=args.compile,
            eval_interval=args.eval_interval,
            tune_threshold_interval=args.tune_threshold_interval,
            evaluate_train_split=False,
            dataloader_num_workers=args.dataloader_workers,
            progress_bar=False,
        )

        # 4. Train, evaluate, and return the metric Optuna needs to maximize
        try:
            summary = train_model(bundle, config, trial_dir)
            trial.set_user_attr("best_validation_rmse", summary["best_validation_rmse"])
            trial.set_user_attr("checkpoint_epoch", summary["checkpoint_epoch"])
            trial.set_user_attr("checkpoint_validation_auc", summary["checkpoint_validation_auc"])
            return summary["best_validation_auc"]
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            trial.set_user_attr("failure_reason", "cuda_oom")
            raise exc

    # Create the study and run the optimization loop
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, catch=(torch.cuda.OutOfMemoryError,))

    print("\n--- Optuna Sweep Finished ---")
    print(f"Best Trial: #{study.best_trial.number} (AUC: {study.best_trial.value:.4f})")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save the best overall parameters to a summary JSON
    best_summary = {
        "study_name": args.study_name,
        "storage": storage_url,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    (output_dir / "optuna_best_results.json").write_text(json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
