# FishCast

This repository now contains a first runnable implementation of the methodology in [FishCastMethodology.md](./FishCastMethodology.md) using the merged dataset in [data/krillcast_merged.csv](./data/krillcast_merged.csv).

See [MODEL_VARIANTS.md](./MODEL_VARIANTS.md) for the current model family.

## What it does

- Loads the merged krill and environmental dataset directly from CSV.
- Normalizes the environmental covariates and spatio-temporal coordinates.
- Creates sector-based hotspot labels using a 90th percentile threshold.
- Initializes 500 inducing points with K-means over `[fractional_year, latitude, longitude]`.
- Trains multiple krill-forecasting model variants, including:
  - `climate_conditioned_gp`: the baseline variational GP
  - `autoregressive_gp`: the same GP plus explicit lagged krill features and recursive rollout support
  - `latent_state_augmented_gp`: the autoregressive GP plus a learned latent ecological state over lagged krill history
  - `autoregressive_sequence`: a GRU-based recursive forecaster with `seq_len` and `horizon_len`
  - `convlstm_baseline`: a grid-based ConvLSTM baseline over yearly sector maps
- The GP variants use:
  - a linear environmental mean module,
  - a 3-layer MLP that produces location-specific spatial covariance matrices,
  - a temporal Gaussian kernel,
  - a shared latent field used for density regression and hotspot classification.

## Run

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py --data data/krillcast_merged.csv --output-dir artifacts/fishcast_run
```

Train the autoregressive GP:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py \
  --model-type autoregressive_gp \
  --lag-density-step 1 \
  --lag-mean-window 3 \
  --lag-hotspot-step 1 \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/fishcast_autoregressive_gp
```

Train the latent-state augmented GP:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py \
  --model-type latent_state_augmented_gp \
  --lag-density-step 1 \
  --lag-mean-window 3 \
  --lag-hotspot-step 1 \
  --latent-state-dim 8 \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/fishcast_latent_state_gp
```

Train the autoregressive sequence model:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py \
  --model-type autoregressive_sequence \
  --seq-len 5 \
  --horizon-len 3 \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/fishcast_autoregressive
```

Train the ConvLSTM baseline:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py \
  --model-type convlstm_baseline \
  --seq-len 5 \
  --horizon-len 3 \
  --autoregressive-hidden-dim 64 \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/fishcast_convlstm
```

For a small classification-weight sweep with scheduler and threshold tuning:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python train_fishcast.py \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/fishcast_sweep \
  --classification-weights 1.0,2.0,5.0 \
  --scheduler-monitor val_rmse \
  --scheduler-patience 5 \
  --threshold-steps 41
```

Artifacts are written to `artifacts/fishcast_run/`:

- `fishcast_model.pt`: trained model weights
- `metrics.json`: training history and summary metrics

For sweep runs, `train_fishcast.py` also writes `sweep_summary.json` in the output directory.

## Forecast Timelapse

Generate an autoregressive yearly rollout to 2050 and render PNG frames every 5 years:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python forecast.py \
  --run-dir artifacts/FishCast_LatentStateGP_v1 \
  --train-data data/krillcast_merged.csv \
  --cmip6-file data/mpi_esm1_2_ssp245_regridded.nc \
  --start-year 2025 \
  --target-year 2050 \
  --frame-step-years 5 \
  --batch-size 25000 \
  --device cpu
```

This writes:

- a final-year forecast CSV
- `frames/forecast_density_<year>.png`
- `frames/forecast_hotspot_probability_<year>.png`
- `timelapse_manifest.json`

Add `--write-frame-data --output-format parquet` if you also want compact per-frame forecast tables alongside the PNGs.

## Visualize

Generate forecast maps and learned kernel ellipses from a trained run:

```bash
/Users/jevontwitty/Documents/GitHub/FishCast/main_fishcast/krillcast/bin/python visualize_fishcast.py \
  --run-dir artifacts/FishCast_Test_v1 \
  --baseline-year 2010 \
  --target-year 2050
```

This writes a `visualizations/` folder inside the run directory containing:

- `forecast_grid_predictions.csv`
- `forecast_density_change.csv` when `--baseline-year` is provided
- `forecast_density.png`
- `forecast_density_change.png` when `--baseline-year` is provided
- `forecast_hotspot_probability.png`
- `kernel_geometry.png`
- `kernel_ellipses.csv`

## Notes

- The merged CSV already includes environmental covariates, so the CMIP6 join described in the draft is treated as precomputed.
- The methodology mentions chlorophyll-a, but this column is not present in the provided CSV. The implementation uses the available merged covariates: `SST`, `SIA`, `SIM`, `SIT`, `SSS`, `SAP`, and `SLP`.
- `gpytorch` is required to run the current model implementation. If your shell is not already inside the `krillcast` virtual environment, use that interpreter path explicitly.
- `metrics.json` now includes tuned hotspot thresholds, per-epoch learning rates, and GP diagnostics such as likelihood noise, kernel output scale, and temporal lengthscale.
- Forecasting and visualization helpers support the GP family directly, including recursive rollout for `autoregressive_gp` and `latent_state_augmented_gp`. The `autoregressive_sequence` model is trained and evaluated through the sequence pipeline, but would still need a dedicated recursive CMIP6 rollout script.
