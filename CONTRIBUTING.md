# Contributing

## Workflow

1. Pick an issue, or open one describing what you want to do. Assign yourself.
2. Create a short-lived branch from `main`: `git checkout -b <subteam>/<short-description>`, for example `analysis/spatial-block-cv` or `viz/leaderboard-page`.
3. Commit small, focused changes with clear messages.
4. Push and open a pull request. Link the issue with `Closes #12`.
5. At least one reviewer approves, ideally someone from another subteam for anything that changes data or evaluation. Your subteam lead merges.

We don't use long-lived personal or team branches; they drift and become hard to merge.

## Rules that protect our results

- **Never tune on the test years.** Validation (1998-2004) picks settings and thresholds. Test (2004-2016) is scored once, at the end.
- **Fit preprocessing on training rows only.** Scaling, imputation and hotspot cut-offs use `fit_index`.
- **Every model result goes through `scripts/benchmark_skill.py`** so it is compared with the same baselines on the same years.
- **Report intervals, not single numbers.** Use at least three seeds.
- **If you change `fishcast/data.py`, bump `PREPROCESSING_VERSION`** and note the change in `docs/`.

## Code

- Python 3.9+, formatted to a 120-character line length. `ruff check .` must pass.
- Add or update a test in `tests/` for any change to data loading or evaluation.
- Don't commit data, model weights, notebooks with large outputs, API keys or credentials.

## Experiments

Record each experiment in `analysis/experiments/YYYY-MM-DD-short-name.md` with the command you ran, the commit hash, the seeds, and the benchmark table it produced.
