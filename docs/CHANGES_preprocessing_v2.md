# Preprocessing v2 — data leakage and correctness fixes (27 Sep 2026)

Originals are in `_backup_before_leakage_fix/`. Runs trained before this change still load:
`load_trained_fishcast` sees no `preprocessing_version` in their `metrics.json` and rebuilds
the data exactly as before (checked: identical tensors). New runs use v2 by default.

## What changed (GP family: `load_krill_dataset`, `training/gp.py`)

1. **Held-out test set.** Chronological split is now train (1926–1998) / validation
   (1998–2004) / test (2004–2016), 8,766 / 1,896 / 1,882 hauls. Validation still picks the
   checkpoint and the hotspot threshold; the test years are scored once at the end.
   `metrics.json` gains `test_metrics` (RMSE, MAE, F1, ROC-AUC, **PR-AUC**,
   `rmse_skill_vs_constant`) and `test_baseline` (constant-prediction RMSE, no-skill PR-AUC).
   Flag: `--test-fraction` (default 0.15).
2. **No validation/test data in preprocessing.** Feature scaling, median fills, lag-feature
   fills, per-sector hotspot cut-offs, inducing points and target stats are fitted on
   training rows only. Sectors with no training hauls use the overall training cut-off.
3. **Sea-ice thickness.** 11,942 of 12,544 hauls have SIT blank because there is no ice
   (SIA = 0). These are now 0; before, they were median-filled to 0.5 m of ice.
4. **Missing-value flags.** `SST_MISSING` (118 hauls with no SST/SIA/SIM/SSS) and
   `SIT_MISSING` (295 hauls with ice but no thickness) are added as 0/1 features.
5. **Lag features were misaligned — every row.** `_add_lag_features` sorted rows by sector
   but coordinates, targets and split indices kept the original order, so in
   `autoregressive_gp` and `latent_state_augmented_gp` each haul's climate + lag features
   came from a different haul. Fixed (row order restored, with an alignment check).
   **All previous lag-model results are invalid**, including `FishCast_LatentStateGP_v1`.
6. **Longitude seam.** The kernel now uses a south-polar projection (`POLAR_X`, `POLAR_Y`,
   degrees from the pole) instead of raw lat/lon, so hauls either side of ±180° are
   neighbours. Flag: `--spatial-coords latlon` restores the old behaviour.
   Kernel-geometry outputs are converted back to lat/lon degrees, so plots are comparable.

7. **Hotspot labels needed krill.** With 36% of hauls catching none, many sectors had a
   90th percentile of 0, so `>= threshold` labelled 609 empty hauls as hotspots (32% of all
   hotspot labels, 566 of them in training years). A hotspot must now contain krill, and
   sectors with fewer than 20 training hauls use the overall training cut-off.

## Also updated
- `forecast.py` and `visualization.py` apply the same sea-ice rule and missing flags to
  CMIP6 grids (`fill_env_features`) and project coordinates the same way.
- The sequence and ConvLSTM loaders get the sea-ice fix only. They still fit scaling and
  hotspot cut-offs on all years and have no test split — follow-up for the Analysis team.

## Retrain before comparing
Any run from before this change is not comparable with v2 runs. Retrain, then compare
models on `test_metrics` against `test_baseline`.


## Found 3 October 2026: longitude bug in the dataset merge

The Colab notebook that built `krillcast_merged.csv` matched KRILLBASE longitudes (-180..180) to a climate grid stored as 0-360 without converting them. Every haul west of Greenwich (77%) got the environmental values of the 0.5 E grid column at its latitude and date. Checked directly: among hauls on the same date and latitude row but at three or more different longitudes, 474 of 610 groups west of Greenwich have identical SST, against 0 of 178 east of Greenwich. This likely explains much of the missing climate-only skill. `scripts/data/build_dataset.py` fixes it; rebuild the file and rerun the benchmark before drawing conclusions about climate predictors.
