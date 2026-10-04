from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    classification_weight: float = 1.0
    seed: int | None = None
    kernel_jitter: float = 1e-3
    hidden_dim: int = 64
    device: str = "cpu"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_monitor: str = "val_rmse"
    scheduler_min_lr: float = 1e-5
    threshold_steps: int = 41
    compile_model: bool = False
    checkpoint_monitor: str = "val_rmse"
    eval_interval: int = 1
    tune_threshold_interval: int = 1
    evaluate_train_split: bool = True
    dataloader_num_workers: int = 0
    progress_bar: bool = True
    model_name: str = "climate_conditioned_gp"
    seq_len: int = 5
    horizon_len: int = 3
    autoregressive_hidden_dim: int = 64
    lag_density_step: int = 1
    lag_mean_window: int = 3
    lag_hotspot_step: int = 1
    latent_state_dim: int = 8
