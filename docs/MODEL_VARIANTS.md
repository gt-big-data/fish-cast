# FishCast Model Variants

This repository now exposes five explicit model families for krill forecasting.

## 1. Climate-Conditioned GP

Model key: `climate_conditioned_gp`

This is the baseline spatio-temporal Gaussian Process model. It conditions on:

- time
- latitude / longitude
- environmental covariates (`SST`, `SIA`, `SIM`, `SIT`, `SSS`, `SAP`, `SLP`)

It does **not** consume explicit lagged krill abundance features. Temporal memory is handled implicitly by the GP kernel.

## 2. Autoregressive GP

Model key: `autoregressive_gp`

This model keeps the same non-stationary GP backbone as the baseline, but augments the mean path with lightweight ecological memory features computed by spatial sector:

- `KRILL_LAG{k}_LOG_DENSITY`
- `KRILL_LAG{w}_LOG_MEAN`
- `HOTSPOT_LAG{h}`

The exact lag settings are configurable from the CLI, and this model is intended to be rolled forward recursively at forecast time:

- `--lag-density-step`
- `--lag-mean-window`
- `--lag-hotspot-step`

This is the minimal upgrade path when we want autoregressive information without abandoning the GP formulation.

## 3. Autoregressive Sequence Forecaster

Model key: `autoregressive_sequence`

This model is a true sequence forecasting pipeline. It first aggregates the krill dataset into sector-year sequences, then trains a GRU encoder-decoder that consumes:

- past environmental covariates
- past krill density
- past hotspot state

and rolls forward recursively over a configurable forecast horizon.

Key hyperparameters:

- `seq_len`: number of historical timesteps used as context
- `horizon_len`: number of future timesteps predicted autoregressively

## 4. Latent-State Augmented GP

Model key: `latent_state_augmented_gp`

This model keeps the full autoregressive GP backbone, but instead of using the lagged krill inputs only as raw extra covariates, it learns a compact latent ecological state from them.

It still uses:

- the deep non-stationary spatial GP
- inducing points
- climate-conditioned mean structure
- recursive rollout at forecast time

But it also adds:

- a learned latent state embedding over the lagged krill density and hotspot history
- a state-informed density path
- a state-informed hotspot path

This is useful when you want explicit ecological memory while still preserving the GP-based spatial field.

## 5. ConvLSTM Baseline

Model key: `convlstm_baseline`

This is a spatial deep-learning baseline built on yearly sector grids instead of pointwise Gaussian Processes. Historical observations are aggregated into a 2D sector lattice for each year, and a ConvLSTM rolls the grid forward through time using:

- past environmental grids
- past krill density grids
- past hotspot grids

It predicts future krill density and hotspot probability maps over the same sector lattice, with masked losses applied only where observations exist.

## Suggested Use

- Use `climate_conditioned_gp` when you want the cleanest direct climate-conditioned baseline.
- Use `autoregressive_gp` when you want autoregressive memory while keeping the spatial GP, inducing points, and deep non-stationary kernel.
- Use `latent_state_augmented_gp` when you want the autoregressive GP plus a learned compact ecological state over the lag history.
- Use `autoregressive_sequence` when you want genuine recursive forecasting behavior and explicit sequence modeling.
- Use `convlstm_baseline` when you want a neural spatio-temporal baseline that explicitly models 2D spatial neighborhoods on aggregated sector grids.
