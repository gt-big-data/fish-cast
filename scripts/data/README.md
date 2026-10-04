# Data pipeline

These scripts rebuild every data file the project uses, replacing the old Colab notebook
(`notebooks/legacy_dataset_merge.ipynb`). Run them from the repository root, in order:

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `download_cmip6.py --experiment historical` | `data/raw/cmip6/historical/<model>/*.nc` |
| 2 | `build_climate_grid.py` | `data/interim/<model>_historical_regridded.nc` |
| 3 | `build_dataset.py` | `data/krillcast_merged.csv` (training data) |
| 4 | `download_cmip6.py --experiment ssp2_4_5 --start 2015 --end 2049` then `build_climate_grid.py` | future climate grid for `scripts/forecast.py` |
| 5 | `convert_mpa_to_wgs84.py` | `data/CCAMLR_MPA_wgs84.geojson` for maps and spillover |

Steps 1 and 4 need a Copernicus CDS API key. Downloads are large and slow; run them on PACE.

**Known issue in the existing `krillcast_merged.csv`:** it was built by the notebook, which
did not convert longitudes before matching to the 0-360 climate grid. For the 77% of hauls
west of Greenwich, the climate values come from the 0.5 E column, not the haul's location.
Rebuilding with `build_dataset.py` fixes this. ERA5 and NSIDC (planned) will get their own
download scripts here.
