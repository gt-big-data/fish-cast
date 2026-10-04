"""Skill benchmark for FishCast: GP models vs. simple baselines on held-out years.

Two tracks, because they answer different questions:
  * nowcast  - may use krill observed earlier in the same 5-degree sector (lag features).
               Skill here does NOT carry to a 2040 projection, where no krill is observed.
  * climate  - climate covariates, location and season only. This is the skill a
               climate-driven projection (and the spillover map) depends on.

Every model uses the same preprocessing-v2 split: train 1926-1998, validation 1998-2004,
test 2004-2016. Validation picks thresholds/checkpoints; test is scored once.
Confidence intervals come from a year-block bootstrap (hauls in the same year resampled
together), so correlated hauls from one cruise do not inflate certainty.

Usage:
  python benchmark_skill.py --gp-runs <run_dir> [<run_dir> ...] --output benchmark_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fishcast.data import add_polar_coords, load_krill_dataset, TARGET_COLUMN

LAG_COLUMNS = ["KRILL_LAG1_LOG_DENSITY", "KRILL_LAG3_LOG_MEAN", "HOTSPOT_LAG1"]


def build_frame(csv: str) -> tuple[pd.DataFrame, dict]:
    bundle = load_krill_dataset(csv, include_lag_features=True)
    frame = add_polar_coords(bundle.dataframe.reset_index(drop=True))
    frame["log_density"] = bundle.target.numpy().ravel()
    frame["hotspot"] = bundle.hotspot.numpy().ravel()
    frame["split"] = "train"
    frame.loc[bundle.val_indices, "split"] = "val"
    frame.loc[bundle.test_indices, "split"] = "test"
    frame["year"] = np.floor(frame["FRACTIONAL_YEAR"]).astype(int)
    season = 2 * np.pi * (frame["FRACTIONAL_YEAR"] % 1.0)
    frame["SEASON_SIN"], frame["SEASON_COS"] = np.sin(season), np.cos(season)
    frame["sector"] = (
        (np.floor(frame["LATITUDE"] / 5) * 5).astype(int).astype(str)
        + "_" + (np.floor(frame["LONGITUDE"] / 5) * 5).astype(int).astype(str)
    )
    return frame, {"bundle": bundle}


def metrics(y, pred, hot, score) -> dict:
    out = {"rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}
    if score is not None and np.unique(hot).size > 1:
        out["roc_auc"] = float(roc_auc_score(hot, score))
        out["pr_auc"] = float(average_precision_score(hot, score))
    return out


def block_bootstrap(test: pd.DataFrame, preds: dict, n: int = 1000, seed: int = 0) -> dict:
    """95% CIs for each model's RMSE skill vs. constant and PR-AUC, resampling years."""
    rng = np.random.default_rng(seed)
    years = test["year"].to_numpy()
    uniq = np.unique(years)
    rows_by_year = {yr: np.flatnonzero(years == yr) for yr in uniq}
    y, hot = test["log_density"].to_numpy(), test["hotspot"].to_numpy()
    const = preds["Constant (train mean)"][0]
    draws = {name: {"skill": [], "pr_auc": []} for name in preds}
    for _ in range(n):
        idx = np.concatenate([rows_by_year[yr] for yr in rng.choice(uniq, size=len(uniq), replace=True)])
        base = np.sqrt(np.mean((y[idx] - const[idx]) ** 2))
        for name, (pred, score) in preds.items():
            draws[name]["skill"].append(1 - np.sqrt(np.mean((y[idx] - pred[idx]) ** 2)) / base)
            if score is not None and np.unique(hot[idx]).size > 1:
                draws[name]["pr_auc"].append(average_precision_score(hot[idx], score[idx]))
    ci = {}
    for name, d in draws.items():
        ci[name] = {
            "skill_ci": [float(np.percentile(d["skill"], 2.5)), float(np.percentile(d["skill"], 97.5))],
            "pr_auc_ci": [float(np.percentile(d["pr_auc"], 2.5)), float(np.percentile(d["pr_auc"], 97.5))] if d["pr_auc"] else None,
        }
    return ci


def fit_catboost(train, val, test, features, seed=0):
    from catboost import CatBoostClassifier, CatBoostRegressor

    reg = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, loss_function="RMSE",
                            random_seed=seed, verbose=False, od_type="Iter", od_wait=100)
    reg.fit(train[features], train["log_density"], eval_set=(val[features], val["log_density"]), use_best_model=True)
    clf = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, loss_function="Logloss",
                             random_seed=seed, verbose=False, od_type="Iter", od_wait=100, auto_class_weights="Balanced")
    clf.fit(train[features], train["hotspot"], eval_set=(val[features], val["hotspot"]), use_best_model=True)
    return reg.predict(test[features]), clf.predict_proba(test[features])[:, 1], dict(zip(features, reg.get_feature_importance()))


def gp_test_predictions(run_dir: str, test_index: np.ndarray):
    from fishcast.visualization import load_trained_fishcast

    model, bundle, run_metrics = load_trained_fishcast(run_dir)
    model.eval(); model.gp_model.eval(); model.likelihood.eval()
    with torch.no_grad():
        out = model(bundle.features[test_index], bundle.coords[test_index])
    uses_lags = any(c in bundle.feature_columns for c in LAG_COLUMNS)
    return out["density"].numpy().ravel(), out["hotspot_probability"].numpy().ravel(), run_metrics.get("model_name"), uses_lags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/krillcast_merged.csv")
    parser.add_argument("--gp-runs", nargs="*", default=[], help="Run dirs trained with preprocessing v2.")
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    frame, _ = build_frame(args.data)
    train, val, test = (frame[frame.split == s] for s in ("train", "val", "test"))
    y, hot = test["log_density"].to_numpy(), test["hotspot"].to_numpy()
    rows, preds, track, importances = [], {}, {}, {}

    def add(name, trk, pred, score):
        preds[name] = (np.asarray(pred, dtype=float), None if score is None else np.asarray(score, dtype=float))
        track[name] = trk

    # --- baselines
    add("Constant (train mean)", "baseline", np.full(len(test), train["log_density"].mean()), None)
    sector_mean = train.groupby("sector")["log_density"].mean()
    sector_hot = train.groupby("sector")["hotspot"].mean()
    add("Sector climatology", "climate",
        test["sector"].map(sector_mean).fillna(train["log_density"].mean()).to_numpy(),
        test["sector"].map(sector_hot).fillna(train["hotspot"].mean()).to_numpy())
    add("Persistence (last haul in sector)", "nowcast",
        test["KRILL_LAG1_LOG_DENSITY"].to_numpy(), test["KRILL_LAG1_LOG_DENSITY"].to_numpy())
    add("Persistence (mean of last 3)", "nowcast",
        test["KRILL_LAG3_LOG_MEAN"].to_numpy(), test["KRILL_LAG3_LOG_MEAN"].to_numpy())

    env = ["SST", "SIA", "SIM", "SIT", "SSS", "SAP", "SLP"]
    climate_features = env + ["LATITUDE", "POLAR_X", "POLAR_Y", "SEASON_SIN", "SEASON_COS"]
    p, s, imp = fit_catboost(train, val, test, climate_features)
    add("CatBoost, climate only", "climate", p, s); importances["CatBoost, climate only"] = imp
    p, s, imp = fit_catboost(train, val, test, climate_features + LAG_COLUMNS)
    add("CatBoost, climate + recent krill", "nowcast", p, s); importances["CatBoost, climate + recent krill"] = imp

    # --- GP runs (grouped by model name; seeds averaged as an ensemble and reported individually)
    test_index = test.index.to_numpy()
    by_name: dict[str, list] = {}
    for run in args.gp_runs:
        pred, score, name, uses_lags = gp_test_predictions(run, test_index)
        by_name.setdefault((name, uses_lags), []).append((run, pred, score))
    for (name, uses_lags), runs in by_name.items():
        label = f"GP {name}"
        trk = "nowcast" if uses_lags else "climate"
        for run, pred, score in runs:
            add(f"{label} [{Path(run).name}]", trk, pred, score)
        if len(runs) > 1:
            add(f"{label} (mean of {len(runs)} seeds)", trk,
                np.mean([r[1] for r in runs], axis=0), np.mean([r[2] for r in runs], axis=0))

    const_rmse = float(np.sqrt(np.mean((y - preds["Constant (train mean)"][0]) ** 2)))
    ci = block_bootstrap(test, preds, n=args.bootstrap)
    for name, (pred, score) in preds.items():
        m = metrics(y, pred, hot, score)
        rows.append({
            "model": name, "track": track[name], "test_rmse": m["rmse"],
            "skill_vs_constant": 1 - m["rmse"] / const_rmse,
            "skill_ci_low": ci[name]["skill_ci"][0], "skill_ci_high": ci[name]["skill_ci"][1],
            "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
            "pr_auc_ci_low": ci[name]["pr_auc_ci"][0] if ci[name]["pr_auc_ci"] else None,
            "pr_auc_ci_high": ci[name]["pr_auc_ci"][1] if ci[name]["pr_auc_ci"] else None,
            "pr_auc_vs_no_skill": (m["pr_auc"] / hot.mean()) if m.get("pr_auc") else None,
        })
    table = pd.DataFrame(rows).sort_values(["track", "skill_vs_constant"], ascending=[True, False])
    table.to_csv(out_dir / "benchmark_table.csv", index=False)
    summary = {
        "test_rows": int(len(test)), "test_years": [int(test.year.min()), int(test.year.max())],
        "n_test_years": int(test.year.nunique()), "hotspot_prevalence": float(hot.mean()),
        "constant_rmse": const_rmse, "feature_importance": importances,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_importance"}, indent=2))


if __name__ == "__main__":
    main()
