"""Build data/krillcast_merged.csv: KRILLBASE hauls matched to monthly climate fields.

Ported from notebooks/legacy_dataset_merge.ipynb (cells 9-12), with one important fix.

THE FIX: the climate grid stores longitude as 0-360 while KRILLBASE uses -180..180.
The notebook matched them with a nearest-neighbour lookup without converting, so every
haul west of Greenwich (77% of them) was matched to the 0.5 E column instead of its own
location. Its environmental values then varied with latitude and date only. This
script converts longitudes before matching and checks the result.

Example:
    python scripts/data/build_dataset.py \
        --krillbase data/raw/krillbase_cleaned.csv \
        --climate data/interim/mpi_esm1_2_lr_historical_regridded.nc \
        --output data/krillcast_merged.csv

The output has the columns the training code expects:
DATE, FRACTIONAL_YEAR, LATITUDE, LONGITUDE, NUMBER_OF_KRILL_UNDER_1M2, SST, SIA, SIM, SIT, SSS, SAP, SLP
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

KEEP = ["DATE", "LATITUDE", "LONGITUDE", "NUMBER_OF_KRILL_UNDER_1M2"]
VARIABLES = ["SST", "SIA", "SIM", "SIT", "SSS", "SAP", "SLP"]


def load_krillbase(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    df = df[KEEP].replace("", pd.NA).dropna(subset=KEEP)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    df.insert(1, "FRACTIONAL_YEAR", df["DATE"].dt.year + (df["DATE"].dt.dayofyear - 1) / 365.25)
    return df


def grid_longitude(lon: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """Express haul longitudes in the grid's convention (0-360 or -180..180)."""
    if np.nanmax(grid_lon) > 180:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


def match(df: pd.DataFrame, ds: xr.Dataset, variables: list[str]) -> pd.DataFrame:
    time_dim = "Date" if "Date" in ds.coords else "time"
    lon = grid_longitude(df["LONGITUDE"].to_numpy(), ds["lon"].values)
    sel = ds[variables].sel(
        {time_dim: xr.DataArray(df["DATE"].to_numpy(), dims="haul"),
         "lat": xr.DataArray(df["LATITUDE"].to_numpy(), dims="haul"),
         "lon": xr.DataArray(lon, dims="haul")},
        method="nearest",
    )
    out = df.copy()
    for v in variables:
        out[v] = sel[v].values
    # Sanity check: matched grid longitudes must sit within one cell of the haul.
    step = float(np.abs(np.diff(ds["lon"].values[:2]))[0])
    got = sel["lon"].values
    gap = np.abs(((got - lon) + 180) % 360 - 180)
    if np.nanmax(gap) > step:
        raise SystemExit(f"Longitude match failed: max distance {np.nanmax(gap):.1f} deg > grid step {step}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--krillbase", required=True, help="Cleaned KRILLBASE CSV (DATE, LATITUDE, LONGITUDE, NUMBER_OF_KRILL_UNDER_1M2)")
    p.add_argument("--climate", required=True, help="Regridded climate .nc from build_climate_grid.py")
    p.add_argument("--output", default="data/krillcast_merged.csv")
    p.add_argument("--variables", nargs="+", default=VARIABLES)
    a = p.parse_args()

    df = load_krillbase(Path(a.krillbase))
    with xr.open_dataset(a.climate) as ds:
        variables = [v for v in a.variables if v in ds.data_vars]
        skipped = sorted(set(a.variables) - set(variables))
        if skipped:
            print(f"warning: not in climate file, left out: {skipped}")
        merged = match(df, ds, variables)
    merged["DATE"] = merged["DATE"].dt.strftime("%Y-%m-%d")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.output, index=False)

    print(f"saved {a.output}: {len(merged):,} hauls, {merged['DATE'].min()} to {merged['DATE'].max()}")
    for v in variables:
        print(f"  {v}: {merged[v].isna().mean():.1%} missing")


if __name__ == "__main__":
    main()
