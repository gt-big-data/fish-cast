# FishCast

FishCast models where Antarctic krill (*Euphausia superba*) are found and how that may change with the climate, especially relative to the marine protected areas (MPAs) managed by CCAMLR. It is a Big Data Big Impact project at Georgia Tech.

The main question for Fall 2026: does the model predict krill better than simple baselines on years it has not seen? See the semester plan for context, and `docs/CHANGES_preprocessing_v2.md` for the September fixes.

## Repository layout

| Folder | What's in it |
| --- | --- |
| `fishcast/` | The Python package: data loading, models (GPs, sequence models), training, plotting helpers |
| `scripts/` | Command-line entry points: train, benchmark, forecast, visualize |
| `scripts/data/` | Data pipeline: download CMIP6, regrid, build `krillcast_merged.csv`, convert MPA boundaries |
| `jobs/` | Slurm job scripts for the PACE cluster |
| `tests/` | Automated tests, run on every pull request |
| `data/` | Not in git. See `data/README.md` for where to get each file |
| `docs/` | Methodology, model variants, change logs |
| `analysis/` | Analysis subteam notes, experiment write-ups, methods research |
| `viz/` | Data Visualization subteam: `prototype/` polar map, `figures/` chart scripts, web app (to come) |
| `notebooks/` | `01_data_tour` and `02_benchmark_results` for new members; the old merge notebook for reference |

## Quick start

```bash
git clone https://github.com/gt-big-data/fish-cast.git
cd fish-cast
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pytest                               # should pass without any data
```

Then download the data (see `data/README.md`) and run:

```bash
# Train one model (about 5 minutes on a laptop CPU with --epochs 10)
python scripts/train_fishcast.py --model-type latent_state_augmented_gp \
    --data data/krillcast_merged.csv --output-dir artifacts/my_first_run --epochs 10

# Compare trained runs against the baselines on the 2004-2016 test years
python scripts/benchmark_skill.py --gp-runs artifacts/my_first_run --output artifacts/benchmark
```

## How we work

- New here? Start with `ONBOARDING.md`.
- Before opening a pull request, read `CONTRIBUTING.md`.
- Tasks live in GitHub Issues on the FishCast Fall 2026 project board. See `TASKS.md`.

## Key results so far

Only models that use recent krill catches currently beat a constant prediction on the test years (about 5% lower error). No climate-only model shows skill yet. All runs from before 27 September 2026 are invalid; see `docs/CHANGES_preprocessing_v2.md`.
