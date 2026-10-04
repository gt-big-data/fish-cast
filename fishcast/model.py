from __future__ import annotations

from fishcast.config import TrainingConfig
from fishcast.data import DatasetBundle, GridSequenceDatasetBundle, SequenceDatasetBundle
from fishcast.models.autoregressive import AutoregressiveKrillForecaster, ConvLSTMKrillForecaster
from fishcast.models.gp import (
    AutoregressiveGPModel,
    ClimateConditionedGPModel,
    CovarianceNetwork,
    DeepNonStationaryKernel,
    FishCastModel,
    LagFeatureGPModel,
    LatentKrillGP,
    LatentStateAugmentedGPModel,
)
from fishcast.training.autoregressive import evaluate_autoregressive_model as _evaluate_autoregressive_model
from fishcast.training.autoregressive import train_autoregressive_model as _train_autoregressive_model
from fishcast.training.autoregressive import train_convlstm_model as _train_convlstm_model
from fishcast.training.gp import evaluate_gp_model as _evaluate
from fishcast.training.gp import extract_gp_diagnostics as _extract_gp_diagnostics
from fishcast.training.gp import train_gp_model as _train_gp_model


def train_model(bundle: DatasetBundle | SequenceDatasetBundle | GridSequenceDatasetBundle, config: TrainingConfig, output_dir: str):
    if config.model_name == "climate_conditioned_gp":
        return _train_gp_model(bundle, config, output_dir, ClimateConditionedGPModel)
    if config.model_name in {"autoregressive_gp", "lag_feature_gp"}:
        return _train_gp_model(bundle, config, output_dir, AutoregressiveGPModel)
    if config.model_name == "latent_state_augmented_gp":
        return _train_gp_model(bundle, config, output_dir, LatentStateAugmentedGPModel)
    if config.model_name == "autoregressive_sequence":
        if not isinstance(bundle, SequenceDatasetBundle):
            raise TypeError("autoregressive_sequence requires a SequenceDatasetBundle.")
        return _train_autoregressive_model(bundle, config, output_dir)
    if config.model_name == "convlstm_baseline":
        if not isinstance(bundle, GridSequenceDatasetBundle):
            raise TypeError("convlstm_baseline requires a GridSequenceDatasetBundle.")
        return _train_convlstm_model(bundle, config, output_dir)
    raise ValueError(f"Unsupported model_name: {config.model_name}")


__all__ = [
    "AutoregressiveKrillForecaster",
    "ConvLSTMKrillForecaster",
    "AutoregressiveGPModel",
    "ClimateConditionedGPModel",
    "CovarianceNetwork",
    "DatasetBundle",
    "DeepNonStationaryKernel",
    "FishCastModel",
    "LagFeatureGPModel",
    "LatentKrillGP",
    "LatentStateAugmentedGPModel",
    "GridSequenceDatasetBundle",
    "SequenceDatasetBundle",
    "TrainingConfig",
    "_evaluate",
    "_evaluate_autoregressive_model",
    "_extract_gp_diagnostics",
    "_train_autoregressive_model",
    "_train_convlstm_model",
    "_train_gp_model",
    "train_model",
]
