"""Audit GP hotspot rankings without changing scores, labels, or fitted settings.

Use explicit --data for corrected datasets. Outputs are diagnostics, not model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from fishcast.data import TARGET_COLUMN, build_sector_keys
from fishcast.visualization import load_trained_fishcast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records, roc_lines, sector_frames = [], [], []
    for run in args.runs:
        model, bundle, saved = load_trained_fishcast(run, data_path=args.data)
        if saved["model_name"] != "climate_conditioned_gp":
            raise ValueError("This diagnostic audits the climate-only GP hotspot head.")
        name = Path(run).name
        scale = float(model.hotspot_scale.detach())
        bias = float(model.hotspot_bias.detach())
        for split, idx in [
            ("train", bundle.train_indices),
            ("validation", bundle.val_indices),
            ("test", bundle.test_indices),
        ]:
            with torch.no_grad():
                result = model(bundle.features[idx], bundle.coords[idx])
            hot = bundle.hotspot[idx].numpy().ravel()
            density = result["density"].numpy().ravel()
            latent = result["latent"].numpy().ravel()
            score = result["hotspot_probability"].numpy().ravel()
            logits = result["hotspot_logits"].numpy().ravel()
            mean_path = density - latent
            residual = bundle.target[idx].numpy().ravel() - mean_path
            # The climate-only head must follow this exact positive-class equation.
            formula_error = float(np.max(np.abs(logits - (scale * latent + bias))))
            rank = np.argsort(score, kind="stable")
            n = max(1, len(idx) // 10)
            record = {
                "run": name,
                "model": saved["model_name"],
                "seed": saved["config"]["seed"],
                "epochs": saved["config"]["epochs"],
                "checkpoint_epoch": saved["checkpoint_epoch"],
                "split": split,
                "rows": len(idx),
                "hotspots": int(hot.sum()),
                "hotspot_scale": scale,
                "hotspot_bias": bias,
                "roc_auc": float(roc_auc_score(hot, score)),
                "inverse_roc_auc_diagnostic_only": float(roc_auc_score(hot, 1 - score)),
                "pr_auc": float(average_precision_score(hot, score)),
                "density_roc_auc": float(roc_auc_score(hot, density)),
                "latent_roc_auc": float(roc_auc_score(hot, latent)),
                "mean_path_roc_auc": float(roc_auc_score(hot, mean_path)),
                "observed_residual_roc_auc": float(roc_auc_score(hot, residual)),
                "latent_observed_residual_spearman": float(spearmanr(latent, residual).statistic),
                "density_hotspot_spearman": float(spearmanr(density, hot).statistic),
                "mean_score_hotspot": float(score[hot == 1].mean()),
                "mean_score_ordinary": float(score[hot == 0].mean()),
                "bottom_decile_hotspot_rate": float(hot[rank[:n]].mean()),
                "top_decile_hotspot_rate": float(hot[rank[-n:]].mean()),
                "climate_head_formula_max_error": formula_error,
            }
            if split == "test":
                raw = bundle.dataframe.iloc[idx].copy()
                assert np.allclose(
                    np.log1p(raw[TARGET_COLUMN].to_numpy()), bundle.target[idx].numpy().ravel(), atol=1e-6
                )
                per_haul = pd.DataFrame(
                    {
                        "row_position": idx,
                        "sector": build_sector_keys(raw, 5, 5).to_numpy(),
                        "hotspot": hot,
                        "probability": score,
                        "latent": latent,
                        "predicted_log_density": density,
                        "year": np.floor(raw["FRACTIONAL_YEAR"].to_numpy()).astype(int),
                    }
                )
                per_haul.to_csv(out / f"{name}_test_predictions.csv", index=False)
                sectors = (
                    per_haul.groupby("sector")
                    .agg(
                        hauls=("hotspot", "size"),
                        observed_rate=("hotspot", "mean"),
                        predicted_probability=("probability", "mean"),
                        latent=("latent", "mean"),
                    )
                    .reset_index()
                )
                train = bundle.dataframe.iloc[bundle.train_indices]
                train_rates = (
                    pd.DataFrame(
                        {
                            "sector": build_sector_keys(train, 5, 5).to_numpy(),
                            "hotspot": bundle.hotspot[bundle.train_indices].numpy().ravel(),
                        }
                    )
                    .groupby("sector")
                    .hotspot.mean()
                )
                sectors["train_observed_rate"] = sectors.sector.map(train_rates)
                sectors["run"] = name
                sector_frames.append(sectors)
                record["sector_score_observed_spearman"] = float(
                    spearmanr(sectors.predicted_probability, sectors.observed_rate).statistic
                )
                populated = sectors[sectors.hauls >= 20]
                record["sectors_with_at_least_20_test_hauls"] = len(populated)
                record["sector_score_observed_spearman_n20"] = float(
                    spearmanr(populated.predicted_probability, populated.observed_rate).statistic
                )
                record["sector_train_test_rate_spearman"] = float(
                    sectors[["train_observed_rate", "observed_rate"]].corr(method="spearman").iloc[0, 1]
                )
                record["score_year_spearman"] = float(spearmanr(score, raw["FRACTIONAL_YEAR"]).statistic)
                for group in ("sector", "year"):
                    weighted_auc, pairs = 0.0, 0
                    for _, block in per_haul.groupby(group):
                        if block.hotspot.nunique() < 2:
                            continue
                        count = int(block.hotspot.sum() * (len(block) - block.hotspot.sum()))
                        weighted_auc += count * roc_auc_score(block.hotspot, block.probability)
                        pairs += count
                    record[f"within_{group}_pair_weighted_auc"] = weighted_auc / pairs if pairs else None
                if saved["model_name"] == "climate_conditioned_gp":
                    # Intervention holds coordinates fixed and changes only normalized climate inputs.
                    probe_idx = idx[:128]
                    with torch.no_grad():
                        original = model(bundle.features[probe_idx], bundle.coords[probe_idx])
                        changed = model(bundle.features[probe_idx] + 1, bundle.coords[probe_idx])
                    record["climate_input_intervention_score_max_change"] = float(
                        (original["hotspot_probability"] - changed["hotspot_probability"]).abs().max()
                    )
                    record["climate_input_intervention_density_max_change"] = float(
                        (original["density"] - changed["density"]).abs().max()
                    )
                # Compare predictions from the benchmark reload with the training summary.
                record["saved_test_auc_difference"] = record["roc_auc"] - saved["test_metrics"]["auc"]
                record["saved_test_rmse_difference"] = (
                    float(np.sqrt(np.mean((bundle.target[idx].numpy().ravel() - density) ** 2)))
                    - saved["test_metrics"]["rmse"]
                )
                fpr, tpr, _ = roc_curve(hot, score)
                roc_lines.append((name, fpr, tpr, record["roc_auc"]))
            records.append(record)
    pd.DataFrame(records).to_csv(out / "ranking_diagnostics.csv", index=False)
    sectors = pd.concat(sector_frames, ignore_index=True)
    sectors.to_csv(out / "sector_diagnostics.csv", index=False)
    provenance = {
        "data": str(Path(args.data).resolve()),
        "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "runs": args.runs,
    }
    (out / "diagnostic_provenance.json").write_text(json.dumps(provenance, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, fpr, tpr, auc in roc_lines:
        label = name.replace("climate_conditioned_gp_", "")
        diagnostic = name == "diagnostic_climate_20_s0"
        if diagnostic:
            label = "20 epochs, s0"
        axes[0].plot(fpr, tpr, linewidth=1, linestyle="--" if diagnostic else "-", label=f"{label}: {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="False positive rate (fraction)",
        ylabel="True positive rate (fraction)",
        title="Climate-only GP: held-out hotspot ROC",
    )
    axes[0].legend(fontsize=7)
    # Seed 0 is fixed in advance, rather than selected for its performance.
    seed0 = sectors[sectors.run == "climate_conditioned_gp_s0"]
    for sparse, marker, color, label in [
        (True, "x", "0.7", "Fewer than 20 test hauls"),
        (False, "o", "0.25", "At least 20 test hauls"),
    ]:
        selected = seed0[(seed0.hauls < 20) == sparse]
        axes[1].scatter(
            selected.predicted_probability,
            selected.observed_rate,
            s=selected.hauls * 0.5 + 8,
            alpha=0.7,
            marker=marker,
            color=color,
            label=label,
        )
    axes[1].legend(fontsize=7)
    axes[1].set(
        xlabel="Mean predicted hotspot probability (fraction)",
        ylabel="Observed hotspot rate (fraction)",
        title="Seed 0: test rates by 5° sector",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Source: checksum-verified krillcast_merged_v2.csv · test 2004–2016 · "
        "sector marker area scales with haul count",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "hotspot_rankings.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
