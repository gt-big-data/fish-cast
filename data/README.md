# Data

Nothing in this folder is committed to git. Get the files below from the team shared drive (ask a Platform lead for the link) and place them here.

| File | Size | What it is |
| --- | --- | --- |
| `krillcast_merged.csv` | 1.7 MB | 12,544 KRILLBASE net hauls (1926-2016) with seven CMIP6 covariates. Required for training and tests on real data |
| `mpi_esm1_2_ssp245_regridded.nc` | 1.5 GB | Future climate projection (MPI-ESM1-2-LR, SSP2-4.5, 2015-2049). Only needed for `forecast.py` |
| `CCAMLR_MPA.json` | 220 KB | Marine protected area boundaries, in EPSG:6932 (metres, south-polar projection) |
| `CCAMLR_MPA_wgs84.geojson` | 190 KB | The same boundaries in longitude/latitude, made by `scripts/data/convert_mpa_to_wgs84.py`; use for maps |

## Original sources

- KRILLBASE: https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/00915
- CMIP6 via the Copernicus Climate Data Store: https://cds.climate.copernicus.eu/datasets/projections-cmip6
- ERA5 (planned): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
- NSIDC sea ice v5 (planned): https://nsidc.org/data/g02202/versions/5

Downloads from the Climate Data Store need a free account and an API key: https://cds.climate.copernicus.eu/how-to-api

Never commit data files, API keys or credentials. The `.gitignore` blocks common data formats, but check `git status` before you commit.

## Known issue: rebuild krillcast_merged.csv

The current file was built by the old notebook, which matched hauls to the 0-360 climate grid without converting longitudes. For the 77% of hauls west of Greenwich, the climate values come from the wrong place. Rebuild it with the pipeline in `scripts/data/` (see its README).

For point-in-polygon tests (is a location inside an MPA?), transform points to EPSG:6932 and test against `CCAMLR_MPA.json`, because three Ross Sea MPAs cross the 180 degree line.
