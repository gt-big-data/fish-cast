import torch

from fishcast import TrainingConfig, load_krill_dataset, train_model
from fishcast.training import gp


def test_cpu_checkpoint_preserves_validation_selected_epoch(synthetic_csv, tmp_path, monkeypatch):
    """Later CPU optimizer steps must not mutate a saved best-epoch snapshot."""
    snapshots = []

    def evaluate(model, bundle, indices, device, **kwargs):
        if kwargs.get("tune_threshold"):
            snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {
            "rmse": float(len(snapshots)),
            "mae": 1.0,
            "f1": 0.1,
            "auc": 0.5,
            "threshold": 0.5,
            "best_f1": 0.1,
        }

    monkeypatch.setattr(gp, "evaluate_gp_model", evaluate)
    monkeypatch.setattr(gp, "evaluate_held_out_test", lambda *args: (None, None))
    bundle = load_krill_dataset(synthetic_csv, num_inducing_points=16)
    config = TrainingConfig(epochs=2, batch_size=128, hidden_dim=8, progress_bar=False, device="cpu")
    train_model(bundle, config, tmp_path)
    checkpoint = torch.load(tmp_path / "fishcast_model.pt", weights_only=True)

    assert checkpoint["best_epoch"] == 1
    assert any(not torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
    for key, value in snapshots[0].items():
        torch.testing.assert_close(checkpoint["model"][key], value, rtol=0, atol=0)
