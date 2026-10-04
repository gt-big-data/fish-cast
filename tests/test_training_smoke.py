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

import subprocess
import json

def test_reproducibility_with_seed(synthetic_csv, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    
    cmd = [
        "python", "-m", "scripts.train_fishcast",
        "--data", str(synthetic_csv),
        "--epochs", "1",
        "--seed", "42",
        "--batch-size", "128",
        "--hidden-dim", "8",
        "--num-inducing-points", "16"
    ]
    
    subprocess.run(cmd + ["--output-dir", str(out1)], check=True)
    subprocess.run(cmd + ["--output-dir", str(out2)], check=True)
    
    with open(out1 / "metrics.json") as f:
        metrics1 = json.load(f)
        
    with open(out2 / "metrics.json") as f:
        metrics2 = json.load(f)
        
    assert metrics1["test_metrics"] == metrics2["test_metrics"]
    assert metrics1["config"]["seed"] == 42
