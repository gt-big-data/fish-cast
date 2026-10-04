"""Combine, clean and regrid one CMIP6 model's monthly fields onto a regular 1-degree grid.

Ported from notebooks/legacy_dataset_merge.ipynb (cells 1-3), with three changes:
paths are arguments, the output keeps only the Southern Ocean by default (--lat-max),
and longitudes are 0-360 to match the original regridded files.

Example:
    python scripts/data/build_climate_grid.py \
        --input data/raw/cmip6/historical/mpi_esm1_2_lr \
        --output data/interim/mpi_esm1_2_lr_historical_regridded.nc

Output variables: SST, SIA, SIM, SIT, SSS (ocean grid) and SAP, SLP (atmosphere grid),
on dims (Date, lat, lon).
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

CMIP6_TO_TARGET = {"tos": "SST", "siconc": "SIA", "simass": "SIM", "sithick": "SIT", "ps": "SAP", "psl": "SLP", "sos": "SSS"}
VALID_RANGES = {
    "SST": (-2.0, 35.0), "SIA": (0.0, 100.0), "SIM": (0.0, 5000.0), "SIT": (0.0, 20.0),
    "SAP": (50_000, 110_000), "SLP": (87_000, 108_600), "SSS": (0.0, 45.0),
}
OCEAN_VARS = ["SST", "SIA", "SIM", "SIT", "SSS"]
ATM_VARS = ["SAP", "SLP"]
FILL = 1e20


def combine(input_dir: Path) -> xr.Dataset:
    parts = []
    for path in sorted(input_dir.rglob("*.nc")):
        ds = xr.open_dataset(path)
        candidates = [v for v in ds.data_vars if ds[v].ndim == 3 and "bnds" not in ds[v].dims]
        if not candidates or candidates[0] not in CMIP6_TO_TARGET:
            print(f"  skip {path.name}: no expected variable")
            continue
        src = candidates[0]
        keep = ds[[src]].rename({src: CMIP6_TO_TARGET[src]})
        for c in ("latitude", "longitude", "lat", "lon"):
            if c in ds and c not in keep.coords:
                keep = keep.assign_coords({c: ds[c]})
        parts.append(keep)
        print(f"  {src} -> {CMIP6_TO_TARGET[src]} ({path.name})")
    if not parts:
        raise SystemExit(f"No usable .nc files under {input_dir}")
    combined = xr.merge(parts, compat="override")
    return combined.rename({"time": "Date"}) if "time" in combined.coords else combined


def clean(ds: xr.Dataset) -> xr.Dataset:
    out = {}
    for var in ds.data_vars:
        da = ds[var].where(np.abs(ds[var]) < FILL * 0.9)
        if var in VALID_RANGES:
            lo, hi = VALID_RANGES[var]
            da = da.where((da >= lo) & (da <= hi))
        out[var] = da
    return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)


def regrid(ds: xr.Dataset, lat_max: float) -> xr.Dataset:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    out_lat = np.arange(-89.5, lat_max, 1.0)
    out_lon = np.arange(0.5, 360.0, 1.0)
    tgt_lat, tgt_lon = np.meshgrid(out_lat, out_lon, indexing="ij")
    xi = np.column_stack([tgt_lat.ravel(), tgt_lon.ravel()])
    result = {}

    ocean = [v for v in OCEAN_VARS if v in ds]
    if ocean:
        lat2d, lon2d = ds["latitude"].values.ravel(), ds["longitude"].values.ravel() % 360
        ok = np.isfinite(lat2d) & np.isfinite(lon2d) & (lat2d < lat_max + 2)
        tri = Delaunay(np.column_stack([lat2d[ok], lon2d[ok]]))
        for var in ocean:
            da = ds[var]
            nt = da.shape[0]
            vals = da.values.reshape(nt, -1)[:, ok].T
            grid = LinearNDInterpolator(tri, vals)(xi).T.reshape(nt, len(out_lat), len(out_lon))
            result[var] = xr.DataArray(grid, dims=["Date", "lat", "lon"],
                                       coords={"Date": ds["Date"], "lat": out_lat, "lon": out_lon}, attrs=da.attrs)
            print(f"  regridded {var}")

    atm = [v for v in ATM_VARS if v in ds]
    if atm:
        lon_name = "lon" if "lon" in ds[atm[0]].dims else "longitude"
        lat_name = "lat" if "lat" in ds[atm[0]].dims else "latitude"
        a = ds[atm].interp({lat_name: out_lat, lon_name: out_lon}, method="linear")
        a = a.rename({lat_name: "lat", lon_name: "lon"}) if lat_name != "lat" else a
        for var in atm:
            result[var] = a[var]
            print(f"  regridded {var}")
    return xr.Dataset(result, attrs=ds.attrs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Folder of downloaded .nc files for ONE model and experiment")
    p.add_argument("--output", required=True, help="Regridded .nc to write")
    p.add_argument("--lat-max", type=float, default=-39.5, help="Keep latitudes south of this (default -39.5)")
    a = p.parse_args()

    print("1/3 combine")
    ds = combine(Path(a.input))
    print("2/3 clean")
    ds = clean(ds)
    print("3/3 regrid")
    out = regrid(ds, a.lat_max)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.output)
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
