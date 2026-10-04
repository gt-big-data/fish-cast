from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fishcast.config import TrainingConfig
from fishcast.data import DEFAULT_ENV_FEATURES, DatasetBundle, lag_feature_column_names
from fishcast.models.gp import ClimateConditionedGPModel, LatentStateAugmentedGPModel

import gpytorch


def compute_best_threshold(hotspot: np.ndarray, hotspot_prob: np.ndarray, steps: int) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, num=steps):
        predictions = (hotspot_prob >= threshold).astype(np.float32)
        score = float(f1_score(hotspot, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def extract_gp_diagnostics(model: ClimateConditionedGPModel) -> Dict[str, float]:
    base_kernel = model.gp_model.covar_module.base_kernel
    diagnostics = {
        "likelihood_noise": float(model.likelihood.noise.detach().cpu().item()),
        "kernel_outputscale": float(model.gp_model.covar_module.outputscale.detach().cpu().item()),
        "temporal_lengthscale": float(base_kernel.temporal_lengthscale.detach().cpu().item()),
        "hotspot_scale": float(model.hotspot_scale.detach().cpu().item()),
        "hotspot_bias": float(model.hotspot_bias.detach().cpu().item()),
    }
    if hasattr(model, "latent_state_dim"):
        diagnostics["latent_state_dim"] = float(model.latent_state_dim)
    return diagnostics


def evaluate_gp_model(
    model: ClimateConditionedGPModel,
    bundle: DatasetBundle,
    indices: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    threshold_steps: int = 41,
    tune_threshold: bool = False,
) -> Dict[str, float]:
    model.eval()
    model.gp_model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        features = bundle.features[indices].to(device)
        coords = bundle.coords[indices].to(device)
        target = bundle.target[indices].cpu().numpy().ravel()
        hotspot = bundle.hotspot[indices].cpu().numpy().ravel()

        outputs = model(features, coords)
        density_pred = outputs["density"].cpu().numpy().ravel()
        hotspot_prob = outputs["hotspot_probability"].cpu().numpy().ravel()
        if tune_threshold:
            threshold, tuned_f1 = compute_best_threshold(hotspot, hotspot_prob, threshold_steps)
        else:
            tuned_f1 = float("nan")
        hotspot_pred = (hotspot_prob >= threshold).astype(np.float32)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(target, density_pred))),
        "mae": float(mean_absolute_error(target, density_pred)),
        "f1": float(f1_score(hotspot, hotspot_pred, zero_division=0)),
        "threshold": float(threshold),
        "best_f1": float(tuned_f1 if tune_threshold else float("nan")),
    }
    metrics["auc"] = float(roc_auc_score(hotspot, hotspot_prob)) if np.unique(hotspot).size > 1 else float("nan")
    # PR-AUC is the more honest ranking score at ~10% hotspot prevalence; no-skill = prevalence.
    metrics["pr_auc"] = float(average_precision_score(hotspot, hotspot_prob)) if np.unique(hotspot).size > 1 else float("nan")
    metrics["hotspot_prevalence"] = float(hotspot.mean()) if hotspot.size else float("nan")
    return metrics


def evaluate_held_out_test(
    model: ClimateConditionedGPModel,
    bundle: DatasetBundle,
    best_state: Dict[str, object],
    device: torch.device,
) -> Tuple[Dict[str, float] | None, Dict[str, float] | None]:
    """Score the selected checkpoint once on the held-out test years, with the
    validation-selected threshold, next to a constant baseline fitted on training rows."""
    test_indices = getattr(bundle, "test_indices", np.array([], dtype=np.int64))
    if len(test_indices) == 0:
        return None, None
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate_gp_model(
        model, bundle, test_indices, device, threshold=float(best_state["best_threshold"]), tune_threshold=False
    )
    train_target = bundle.target[bundle.train_indices].cpu().numpy().ravel()
    test_target = bundle.target[test_indices].cpu().numpy().ravel()
    constant = float(train_target.mean())
    constant_rmse = float(np.sqrt(np.mean((test_target - constant) ** 2)))
    baseline = {
        "constant_prediction": constant,
        "constant_rmse": constant_rmse,
        "constant_mae": float(np.mean(np.abs(test_target - constant))),
        "no_skill_pr_auc": test_metrics["hotspot_prevalence"],
        "no_skill_auc": 0.5,
    }
    test_metrics["rmse_skill_vs_constant"] = 1.0 - test_metrics["rmse"] / constant_rmse if constant_rmse > 0 else float("nan")
    return test_metrics, baseline


def train_gp_model(
    bundle: DatasetBundle,
    config: TrainingConfig,
    output_dir: str | Path,
    model_class: type[ClimateConditionedGPModel],
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    if model_class is LatentStateAugmentedGPModel:
        lag_columns = lag_feature_column_names(
            config.lag_density_step,
            config.lag_mean_window,
            config.lag_hotspot_step,
        )
        # Environmental columns plus their missing-value flags; everything that is not a lag feature.
        env_feature_count = len([column for column in bundle.feature_columns if column not in lag_columns])
        lag_feature_count = len([column for column in bundle.feature_columns if column in lag_columns])
        model = model_class(
            num_features=bundle.features.shape[1],
            inducing_points=bundle.inducing_points.to(device),
            hidden_dim=config.hidden_dim,
            kernel_jitter=config.kernel_jitter,
            env_feature_count=env_feature_count,
            lag_feature_count=lag_feature_count,
            latent_state_dim=config.latent_state_dim,
        ).to(device)
    else:
        model = model_class(
            num_features=bundle.features.shape[1],
            inducing_points=bundle.inducing_points.to(device),
            hidden_dim=config.hidden_dim,
            kernel_jitter=config.kernel_jitter,
        ).to(device)

    if config.compile_model:
        model = torch.compile(model)

    if config.eval_interval < 1:
        raise ValueError("eval_interval must be >= 1.")
    if config.tune_threshold_interval < 1:
        raise ValueError("tune_threshold_interval must be >= 1.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config.scheduler_monitor != "val_auc" else "max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_model, num_data=len(bundle.train_indices))
    train_hotspot = bundle.hotspot[bundle.train_indices]
    positive_count = float(train_hotspot.sum().item())
    negative_count = float(len(train_hotspot) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = TensorDataset(
        bundle.features[bundle.train_indices],
        bundle.coords[bundle.train_indices],
        bundle.target[bundle.train_indices],
        bundle.hotspot[bundle.train_indices],
    )
    use_cuda_loader = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.dataloader_num_workers,
        "pin_memory": use_cuda_loader,
    }
    if config.dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    history: List[Dict[str, float]] = []
    best_val_rmse = float("inf")
    best_val_auc = float("-inf")
    best_state = None
    best_threshold = 0.5

    if config.checkpoint_monitor not in {"val_rmse", "val_auc"}:
        raise ValueError("checkpoint_monitor must be either 'val_rmse' or 'val_auc'.")

    epoch_iterator = range(1, config.epochs + 1)
    if config.progress_bar:
        epoch_iterator = tqdm(epoch_iterator, desc="Training Epochs")
    for epoch in epoch_iterator:
        model.train()
        model.gp_model.train()
        model.likelihood.train()
        regression_total = 0.0
        classification_total = 0.0

        for features, coords, target, hotspot in train_loader:
            features = features.to(device)
            coords = coords.to(device)
            target = target.to(device)
            hotspot = hotspot.to(device)

            outputs = model(features, coords)
            latent_dist = outputs["latent_dist"]
            latent_mean = outputs["latent"]
            baseline = outputs["density"] - latent_mean
            hotspot_logits = outputs["hotspot_logits"]
            residual_target = (target - baseline).squeeze(-1)

            regression = -mll(latent_dist, residual_target)
            classification = bce_loss(hotspot_logits, hotspot)
            loss = regression + config.classification_weight * classification

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            regression_total += float(regression.item()) * len(features)
            classification_total += float(classification.item()) * len(features)

        should_evaluate = (epoch % config.eval_interval == 0) or (epoch == config.epochs)
        should_tune_threshold = ((epoch % config.tune_threshold_interval) == 0) or (epoch == config.epochs)

        if should_evaluate:
            val_metrics = evaluate_gp_model(
                model,
                bundle,
                bundle.val_indices,
                device,
                threshold=best_threshold,
                threshold_steps=config.threshold_steps,
                tune_threshold=should_tune_threshold,
            )
            best_threshold = val_metrics["threshold"]
        else:
            val_metrics = {"rmse": float("nan"), "mae": float("nan"), "f1": float("nan"), "auc": float("nan"), "best_f1": float("nan"), "threshold": float(best_threshold)}

        if config.evaluate_train_split and should_evaluate:
            train_metrics = evaluate_gp_model(
                model,
                bundle,
                bundle.train_indices,
                device,
                threshold=best_threshold,
                threshold_steps=config.threshold_steps,
                tune_threshold=False,
            )
        else:
            train_metrics = {"rmse": float("nan"), "mae": float("nan"), "f1": float("nan"), "auc": float("nan"), "threshold": float(best_threshold)}

        diagnostics = extract_gp_diagnostics(model)
        epoch_record = {
            "epoch": float(epoch),
            "train_regression_loss": regression_total / len(bundle.train_indices),
            "train_classification_loss": classification_total / len(bundle.train_indices),
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "train_threshold_used": train_metrics["threshold"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_best_f1": val_metrics["best_f1"],
            "val_best_threshold": val_metrics["threshold"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **diagnostics,
        }
        history.append(epoch_record)

        if config.progress_bar:
            postfix = {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if should_evaluate:
                postfix["val_rmse"] = f"{val_metrics['rmse']:.4f}"
                postfix["val_auc"] = f"{val_metrics['auc']:.4f}"
            epoch_iterator.set_postfix(postfix)

        if should_evaluate:
            monitor_value = val_metrics["rmse"] if config.scheduler_monitor == "val_rmse" else val_metrics["auc"]
            scheduler.step(monitor_value)

            previous_best_rmse = best_val_rmse
            previous_best_auc = best_val_auc
            best_val_rmse = min(best_val_rmse, val_metrics["rmse"])
            best_val_auc = max(best_val_auc, val_metrics["auc"])

            is_better_checkpoint = (
                val_metrics["rmse"] < previous_best_rmse if config.checkpoint_monitor == "val_rmse"
                else val_metrics["auc"] > (best_state["best_validation_auc"] if best_state is not None else previous_best_auc)
            )

            if is_better_checkpoint:
                best_state = {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "likelihood": {key: value.detach().cpu() for key, value in model.likelihood.state_dict().items()},
                    "best_threshold": best_threshold,
                    "best_validation_rmse": val_metrics["rmse"],
                    "best_validation_auc": val_metrics["auc"],
                    "checkpoint_monitor": config.checkpoint_monitor,
                    "best_epoch": epoch,
                }

    if best_state is None:
        best_state = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "likelihood": {key: value.detach().cpu() for key, value in model.likelihood.state_dict().items()},
            "best_threshold": best_threshold,
            "best_validation_rmse": history[-1]["val_rmse"] if history else float("nan"),
            "best_validation_auc": history[-1]["val_auc"] if history else float("nan"),
            "checkpoint_monitor": config.checkpoint_monitor,
            "best_epoch": history[-1]["epoch"] if history else -1,
        }
    torch.save(best_state, output_dir / "fishcast_model.pt")
    test_metrics, test_baseline = evaluate_held_out_test(model, bundle, best_state, device)

    summary = {
        "config": asdict(config),
        "model_name": config.model_name,
        "data": bundle.metadata,
        "feature_columns": bundle.feature_columns,
        "target_stats": bundle.target_stats,
        "best_validation_rmse": best_val_rmse,
        "best_validation_auc": best_val_auc,
        "checkpoint_monitor": config.checkpoint_monitor,
        "checkpoint_epoch": best_state["best_epoch"],
        "checkpoint_validation_rmse": best_state["best_validation_rmse"],
        "checkpoint_validation_auc": best_state["best_validation_auc"],
        "selected_validation_threshold": best_state["best_threshold"],
        "test_metrics": test_metrics,
        "test_baseline": test_baseline,
        "final_gp_diagnostics": extract_gp_diagnostics(model),
        "history": history,
        "backend": "gpytorch_variational_gp",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
