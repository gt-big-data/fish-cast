# Notebooks

| Notebook | Who it's for | What it does |
| --- | --- | --- |
| `01_data_tour.ipynb` | Everyone, week 1 | Tour of the training data: hauls, zero catches, hotspots, the year-based split |
| `02_benchmark_results.ipynb` | Analysis, Data Viz | How to read `benchmark_table.csv`, check the skill bar, draw the leaderboard |
| `legacy_dataset_merge.ipynb` | Reference only | The original Colab notebook that built `krillcast_merged.csv`. Replaced by `scripts/data/`; it has a longitude bug (see `scripts/data/README.md`) |

Run notebooks from the repository root (`jupyter lab` there) so the `data/` paths work.
Clear outputs before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.
New exploratory notebooks: name them `NN_short_topic.ipynb` and move anything reusable into `fishcast/` or `scripts/`.
