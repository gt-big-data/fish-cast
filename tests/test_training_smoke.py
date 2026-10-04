"""One-epoch training run on synthetic data: checks the pipeline end to end and that test metrics are written."""
from fishcast import TrainingConfig, load_krill_dataset, train_model


def test_gp_trains_and_reports_test_metrics(synthetic_csv, tmp_path):
    bundle = load_krill_dataset(synthetic_csv, num_inducing_points=16, include_lag_features=True)
    config = TrainingConfig(epochs=1, batch_size=128, hidden_dim=8, progress_bar=False,
                            model_name="latent_state_augmented_gp", latent_state_dim=4)
    summary = train_model(bundle=bundle, config=config, output_dir=tmp_path / "run")
    assert (tmp_path / "run" / "metrics.json").exists()
    assert summary["test_metrics"] is not None
    assert "rmse_skill_vs_constant" in summary["test_metrics"]
