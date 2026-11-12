# Data

Nothing in this folder is committed to git. Get the files below from the FishCast Data folder on Google Drive and place them here:
https://drive.google.com/drive/folders/1bZsC04b6aWKsyWe3-XNH_PEvlAaAhTcC

| File | Size | What it is |
| --- | --- | --- |
| `krillcast_merged.csv` | 1.7 MB | 12,544 KRILLBASE net hauls (1926-2016) with seven CMIP6 covariates. Required for training and tests on real data |
| `krillcast_merged_v2.csv` | 1.7 MB | 12,544 KRILLBASE net hauls (1926–2016) with corrected longitude matching. Historical CMIP6 through 2014 and SSP2-4.5 for 2015–2016 |
| `mpi_esm1_2_ssp245_regridded.nc` | 1.5 GB | Future climate projection (MPI-ESM1-2-LR, SSP2-4.5, 2015-2049). Only needed for `forecast.py` |
| `CCAMLR_MPA.json` | 220 KB | Marine protected area boundaries, in EPSG:6932 (metres, south-polar projection) |
| `CCAMLR_MPA_wgs84.geojson` | 190 KB | The same boundaries in longitude/latitude, made by `scripts/data/convert_mpa_to_wgs84.py`; use for maps |

## Checksums (SHA-256)

Check your download matches: `sha256sum <file>` (macOS/Linux, or `shasum -a 256`) or `certutil -hashfile <file> SHA256` (Windows).

| File | SHA-256 |
| --- | --- |
| `krillcast_merged.csv` (v1) | `74c5cbc57ca3113df1ae159a0ae1dc66d025acfad850aa07f0635e6bffb8b390` |
| `krillcast_merged_v2.csv` (v2) | `740eeba5a1d3162aeb92ec30a3145fad3ed5bfc67c42141583e9f175455cc7fa` |
| `mpi_esm1_2_ssp245_regridded.nc` | `bde765659ac2ffae478222e2e2e80b8bd8c3c196e199ae478a5779f77247e31b` |
| `CCAMLR_MPA.json` | `bc0848edb05090dfe4c0c2833ea5a5658f662d6e592bbcafb642e4e1af1bbd4a` |
| `CCAMLR_MPA_wgs84.geojson` | `e3fa68566eff7d3a95efec325d7cb9319c7345c39d9ff9fccaf7acf39b04fd99` |

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
