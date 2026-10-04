"""FishCast package."""

from .data import DatasetBundle, GridSequenceDatasetBundle, SequenceDatasetBundle, load_autoregressive_dataset, load_convlstm_dataset, load_krill_dataset
from .config import TrainingConfig
from .model import (
    AutoregressiveKrillForecaster,
    AutoregressiveGPModel,
    ClimateConditionedGPModel,
    ConvLSTMKrillForecaster,
    FishCastModel,
    LagFeatureGPModel,
    LatentStateAugmentedGPModel,
    train_model,
)
from .visualization import (
    build_forecast_grid,
    load_trained_fishcast,
    predict_grid,
    save_forecast_visualizations,
    save_kernel_visualization,
)

__all__ = [
    "DatasetBundle",
    "SequenceDatasetBundle",
    "GridSequenceDatasetBundle",
    "AutoregressiveKrillForecaster",
    "ConvLSTMKrillForecaster",
    "AutoregressiveGPModel",
    "ClimateConditionedGPModel",
    "FishCastModel",
    "LagFeatureGPModel",
    "LatentStateAugmentedGPModel",
    "TrainingConfig",
    "build_forecast_grid",
    "load_autoregressive_dataset",
    "load_convlstm_dataset",
    "load_krill_dataset",
    "load_trained_fishcast",
    "predict_grid",
    "save_forecast_visualizations",
    "save_kernel_visualization",
    "train_model",
]
