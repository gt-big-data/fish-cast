# Week 1 starter issues

Each issue below is small, has a clear finish line, and gets a new member touching the real code. Create them, and every other semester task, with `python scripts/setup_github.py` (see `TASKS.md`).

## Platform

### 1. Move shared data to the team drive and document it
Put `krillcast_merged.csv`, `CCAMLR_MPA.json` and the regridded CMIP6 file in one shared folder. Add the link and a SHA-256 checksum for each file to `data/README.md`.
**Done when:** a new member can follow `data/README.md` and verify their files match.

### 2. Get the benchmark running on PACE
Adapt `jobs/run_v4_sweep.sbatch` into `jobs/benchmark.sbatch`: train three GP models with five seeds each, then run `scripts/benchmark_skill.py`.
**Done when:** one `sbatch` command produces `benchmark_table.csv` in a dated folder.

### 3. Script the dataset merge
Turn the CMIP6-to-KRILLBASE merge in `notebooks/legacy_dataset_merge.ipynb` into `scripts/build_dataset.py` with no hard-coded Google Drive paths.
**Done when:** the script reproduces `krillcast_merged.csv` (same rows, values within 1e-6).

## Analysis

### 4. Add a seed option to training
`train_fishcast.py` has no `--seed` flag, so runs aren't reproducible. Add one that seeds Python, NumPy and PyTorch, and save it in `metrics.json`.
**Done when:** two runs with the same seed give identical `test_metrics`.

### 5. Why does the climate-only GP rank hotspots backwards?
Its test ROC-AUC is about 0.35. Check the sign of `hotspot_scale`, compare predicted probability with observed hotspot rate by sector, and write up what you find.
**Done when:** a note in `analysis/experiments/` with a likely explanation or ruled-out causes.

### 6. Methods research kickoff (optional)
Read Elith et al. (2008) on boosted regression trees and the sdmTMB tutorials. Write one page on which approach looks most promising for our data and why.
**Done when:** `analysis/methods/initial-survey.md` is merged.

## Data Visualization

### 7. South-polar projection prototype
A single HTML page that draws the Antarctic coastline and CCAMLR MPA boundaries (`data/CCAMLR_MPA.json`) in a south-polar stereographic projection with D3.
**Done when:** the page renders and the MPAs line up with the coastline.

### 8. App skeleton and preview deploys
Set up `viz/app/` with the chosen framework, a lint step in CI, and preview deployments for each pull request (for example GitHub Pages or Vercel).
**Done when:** opening a pull request produces a preview link.

### 9. Leaderboard from benchmark output
A small script that reads `benchmark_table.csv` and produces a chart of each model's skill with its 95% interval, following the style guide.
**Done when:** `python viz/figures/leaderboard.py <table>` writes a PNG and SVG.

### 10. Style guide draft
Write `viz/STYLE_GUIDE.md`: projection, colour maps (perceptually uniform, colour-blind safe), fonts, units and caption rules.
**Done when:** reviewed by one person from each other subteam.
