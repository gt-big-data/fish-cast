from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fishcast.config import TrainingConfig
from fishcast.data import GridSequenceDatasetBundle, SequenceDatasetBundle
from fishcast.models.autoregressive import AutoregressiveKrillForecaster, ConvLSTMKrillForecaster


def compute_best_sequence_threshold(truth: np.ndarray, probs: np.ndarray, steps: int) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, num=steps):
        score = float(f1_score(truth, (probs >= threshold).astype(np.float32), zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_autoregressive_model(
    model: AutoregressiveKrillForecaster,
    bundle: SequenceDatasetBundle,
    indices: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    threshold_steps: int = 41,
    tune_threshold: bool = False,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        past_inputs = bundle.past_inputs[indices].to(device)
        future_env = bundle.future_env[indices].to(device)
        outputs = model(past_inputs, future_env)
        density_pred = outputs["density"].detach().cpu().numpy().ravel()
        hotspot_prob = outputs["hotspot_probability"].detach().cpu().numpy().ravel()
        truth_density = bundle.future_target[indices].cpu().numpy().ravel()
        truth_hotspot = bundle.future_hotspot[indices].cpu().numpy().ravel()
        if tune_threshold:
            threshold, tuned_f1 = compute_best_sequence_threshold(truth_hotspot, hotspot_prob, threshold_steps)
        else:
            tuned_f1 = float("nan")
        hotspot_pred = (hotspot_prob >= threshold).astype(np.float32)

    return {
        "rmse": float(np.sqrt(mean_squared_error(truth_density, density_pred))),
        "mae": float(mean_absolute_error(truth_density, density_pred)),
        "f1": float(f1_score(truth_hotspot, hotspot_pred, zero_division=0)),
        "auc": float(roc_auc_score(truth_hotspot, hotspot_prob)) if np.unique(truth_hotspot).size > 1 else float("nan"),
        "threshold": threshold,
        "best_f1": tuned_f1,
    }


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target).pow(2) * mask
    return squared_error.sum() / mask.sum().clamp_min(1.0)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    per_element = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    masked = per_element * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def evaluate_convlstm_model(
    model: ConvLSTMKrillForecaster,
    bundle: GridSequenceDatasetBundle,
    indices: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    threshold_steps: int = 41,
    tune_threshold: bool = False,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        past_inputs = bundle.past_inputs[indices].to(device)
        future_env = bundle.future_env[indices].to(device)
        future_mask = bundle.future_mask[indices].cpu().numpy().ravel().astype(bool)
        outputs = model(past_inputs, future_env)
        density_pred = outputs["density"].detach().cpu().numpy().ravel()[future_mask]
        hotspot_prob = outputs["hotspot_probability"].detach().cpu().numpy().ravel()[future_mask]
        truth_density = bundle.future_target[indices].cpu().numpy().ravel()[future_mask]
        truth_hotspot = bundle.future_hotspot[indices].cpu().numpy().ravel()[future_mask]
        if tune_threshold:
            threshold, tuned_f1 = compute_best_sequence_threshold(truth_hotspot, hotspot_prob, threshold_steps)
        else:
            tuned_f1 = float("nan")
        hotspot_pred = (hotspot_prob >= threshold).astype(np.float32)

    return {
        "rmse": float(np.sqrt(mean_squared_error(truth_density, density_pred))),
        "mae": float(mean_absolute_error(truth_density, density_pred)),
        "f1": float(f1_score(truth_hotspot, hotspot_pred, zero_division=0)),
        "auc": float(roc_auc_score(truth_hotspot, hotspot_prob)) if np.unique(truth_hotspot).size > 1 else float("nan"),
        "threshold": threshold,
        "best_f1": tuned_f1,
    }


def train_autoregressive_model(bundle: SequenceDatasetBundle, config: TrainingConfig, output_dir: str | Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = AutoregressiveKrillForecaster(
        env_dim=len(bundle.feature_columns),
        hidden_dim=config.autoregressive_hidden_dim,
        horizon_len=config.horizon_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config.scheduler_monitor != "val_auc" else "max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )
    mse_loss = nn.MSELoss()
    positive_count = float(bundle.future_hotspot[bundle.train_indices].sum().item())
    negative_count = float(bundle.future_hotspot[bundle.train_indices].numel() - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = TensorDataset(
        bundle.past_inputs[bundle.train_indices],
        bundle.future_env[bundle.train_indices],
        bundle.future_target[bundle.train_indices],
        bundle.future_hotspot[bundle.train_indices],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )

    history: List[Dict[str, float]] = []
    best_val_rmse = float("inf")
    best_val_auc = float("-inf")
    best_threshold = 0.5
    best_state = None
    epoch_iterator = range(1, config.epochs + 1)
    if config.progress_bar:
        epoch_iterator = tqdm(epoch_iterator, desc="AR Training Epochs")

    for epoch in epoch_iterator:
        model.train()
        regression_total = 0.0
        classification_total = 0.0
        for past_inputs, future_env, future_target, future_hotspot in train_loader:
            past_inputs = past_inputs.to(device)
            future_env = future_env.to(device)
            future_target = future_target.to(device)
            future_hotspot = future_hotspot.to(device)

            outputs = model(past_inputs, future_env)
            regression = mse_loss(outputs["density"], future_target)
            classification = bce_loss(outputs["hotspot_logits"], future_hotspot)
            loss = regression + config.classification_weight * classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            regression_total += float(regression.item()) * len(past_inputs)
            classification_total += float(classification.item()) * len(past_inputs)

        val_metrics = evaluate_autoregressive_model(
            model, bundle, bundle.val_indices, device, threshold=best_threshold, threshold_steps=config.threshold_steps, tune_threshold=True
        )
        best_threshold = val_metrics["threshold"]
        train_metrics = evaluate_autoregressive_model(
            model, bundle, bundle.train_indices, device, threshold=best_threshold, threshold_steps=config.threshold_steps, tune_threshold=False
        ) if config.evaluate_train_split else {
            "rmse": float("nan"),
            "mae": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "threshold": best_threshold,
            "best_f1": float("nan"),
        }

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
        }
        history.append(epoch_record)
        if config.progress_bar:
            epoch_iterator.set_postfix(val_rmse=f"{val_metrics['rmse']:.4f}", val_auc=f"{val_metrics['auc']:.4f}")

        scheduler.step(val_metrics["rmse"] if config.scheduler_monitor == "val_rmse" else val_metrics["auc"])
        previous_best_rmse = best_val_rmse
        previous_best_auc = best_val_auc
        best_val_rmse = min(best_val_rmse, val_metrics["rmse"])
        best_val_auc = max(best_val_auc, val_metrics["auc"])
        is_better = val_metrics["rmse"] < previous_best_rmse if config.checkpoint_monitor == "val_rmse" else val_metrics["auc"] > previous_best_auc
        if is_better:
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "best_threshold": best_threshold,
                "best_validation_rmse": val_metrics["rmse"],
                "best_validation_auc": val_metrics["auc"],
                "best_epoch": epoch,
                "checkpoint_monitor": config.checkpoint_monitor,
            }

    if best_state is None:
        best_state = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "best_threshold": best_threshold,
            "best_validation_rmse": history[-1]["val_rmse"],
            "best_validation_auc": history[-1]["val_auc"],
            "best_epoch": history[-1]["epoch"],
            "checkpoint_monitor": config.checkpoint_monitor,
        }

    torch.save(best_state, output_dir / "fishcast_model.pt")
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
        "history": history,
        "backend": "gru_autoregressive_sequence",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_convlstm_model(bundle: GridSequenceDatasetBundle, config: TrainingConfig, output_dir: str | Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    hidden_dim = max(16, config.autoregressive_hidden_dim // 2)
    model = ConvLSTMKrillForecaster(
        env_dim=len(bundle.feature_columns),
        hidden_dim=hidden_dim,
        horizon_len=config.horizon_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config.scheduler_monitor != "val_auc" else "max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )
    observed_hotspot = bundle.future_hotspot[bundle.train_indices]
    observed_mask = bundle.future_mask[bundle.train_indices]
    positive_count = float((observed_hotspot * observed_mask).sum().item())
    negative_count = float(observed_mask.sum().item() - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)

    train_dataset = TensorDataset(
        bundle.past_inputs[bundle.train_indices],
        bundle.future_env[bundle.train_indices],
        bundle.future_target[bundle.train_indices],
        bundle.future_hotspot[bundle.train_indices],
        bundle.future_mask[bundle.train_indices],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )

    history: List[Dict[str, float]] = []
    best_val_rmse = float("inf")
    best_val_auc = float("-inf")
    best_threshold = 0.5
    best_state = None
    epoch_iterator = range(1, config.epochs + 1)
    if config.progress_bar:
        epoch_iterator = tqdm(epoch_iterator, desc="ConvLSTM Training Epochs")

    for epoch in epoch_iterator:
        model.train()
        regression_total = 0.0
        classification_total = 0.0
        for past_inputs, future_env, future_target, future_hotspot, future_mask in train_loader:
            past_inputs = past_inputs.to(device)
            future_env = future_env.to(device)
            future_target = future_target.to(device)
            future_hotspot = future_hotspot.to(device)
            future_mask = future_mask.to(device)

            outputs = model(past_inputs, future_env)
            regression = _masked_mse(outputs["density"], future_target, future_mask)
            classification = _masked_bce(outputs["hotspot_logits"], future_hotspot, future_mask, pos_weight)
            loss = regression + config.classification_weight * classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            regression_total += float(regression.item()) * len(past_inputs)
            classification_total += float(classification.item()) * len(past_inputs)

        val_metrics = evaluate_convlstm_model(
            model, bundle, bundle.val_indices, device, threshold=best_threshold, threshold_steps=config.threshold_steps, tune_threshold=True
        )
        best_threshold = val_metrics["threshold"]
        train_metrics = evaluate_convlstm_model(
            model, bundle, bundle.train_indices, device, threshold=best_threshold, threshold_steps=config.threshold_steps, tune_threshold=False
        ) if config.evaluate_train_split else {
            "rmse": float("nan"),
            "mae": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "threshold": best_threshold,
            "best_f1": float("nan"),
        }

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
        }
        history.append(epoch_record)
        if config.progress_bar:
            epoch_iterator.set_postfix(val_rmse=f"{val_metrics['rmse']:.4f}", val_auc=f"{val_metrics['auc']:.4f}")

        scheduler.step(val_metrics["rmse"] if config.scheduler_monitor == "val_rmse" else val_metrics["auc"])
        previous_best_rmse = best_val_rmse
        previous_best_auc = best_val_auc
        best_val_rmse = min(best_val_rmse, val_metrics["rmse"])
        best_val_auc = max(best_val_auc, val_metrics["auc"])
        is_better = val_metrics["rmse"] < previous_best_rmse if config.checkpoint_monitor == "val_rmse" else val_metrics["auc"] > previous_best_auc
        if is_better:
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "best_threshold": best_threshold,
                "best_validation_rmse": val_metrics["rmse"],
                "best_validation_auc": val_metrics["auc"],
                "best_epoch": epoch,
                "checkpoint_monitor": config.checkpoint_monitor,
            }

    if best_state is None:
        best_state = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "best_threshold": best_threshold,
            "best_validation_rmse": history[-1]["val_rmse"],
            "best_validation_auc": history[-1]["val_auc"],
            "best_epoch": history[-1]["epoch"],
            "checkpoint_monitor": config.checkpoint_monitor,
        }

    torch.save(best_state, output_dir / "fishcast_model.pt")
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
        "history": history,
        "backend": "convlstm_autoregressive_grid",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
