from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from scripts.benchmark_skill import gp_test_predictions


def test_gp_predictions_load_explicit_dataset(monkeypatch):
    """Benchmark --data must also select the data used to reload GP checkpoints."""
    model = Mock(
        return_value={
            "density": torch.tensor([[2.0]]),
            "hotspot_probability": torch.tensor([[0.8]]),
        }
    )
    bundle = SimpleNamespace(
        features=torch.tensor([[1.0], [3.0]]),
        coords=torch.tensor([[2.0], [4.0]]),
        feature_columns=["SST"],
    )
    loader = Mock(return_value=(model, bundle, {"model_name": "climate_conditioned_gp"}))
    monkeypatch.setattr("fishcast.visualization.load_trained_fishcast", loader)

    pred, score, name, uses_lags = gp_test_predictions("run", np.array([1]), "data/v2.csv")

    loader.assert_called_once_with("run", data_path="data/v2.csv")
    np.testing.assert_array_equal(model.call_args.args[0], [[3.0]])
    np.testing.assert_array_equal(model.call_args.args[1], [[4.0]])
    np.testing.assert_array_equal(pred, [2.0])
    np.testing.assert_allclose(score, [0.8])
    assert name == "climate_conditioned_gp"
    assert not uses_lags
