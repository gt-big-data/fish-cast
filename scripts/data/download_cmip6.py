"""Download monthly CMIP6 fields from the Copernicus Climate Data Store.

Ported from notebooks/legacy_dataset_merge.ipynb (cells 4-5).

Needs a free CDS account and API key: https://cds.climate.copernicus.eu/how-to-api

Examples:
    # Historical runs used to build the training covariates (1926-2014)
    python scripts/data/download_cmip6.py --experiment historical --start 1926 --end 2014

    # Future scenario for projections
    python scripts/data/download_cmip6.py --experiment ssp2_4_5 --start 2015 --end 2049 --models mpi_esm1_2_lr

Files land in <out>/<experiment>/<model>/<variable>.nc. Existing files are skipped,
so an interrupted download can simply be rerun.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

DATASET = "projections-cmip6"
MODELS = ["ec_earth3_veg_lr", "hadgem3_gc31_ll", "mpi_esm1_2_lr"]
VARIABLES = {
    "sea_surface_temperature": "SST",
    "sea_ice_area_percentage_on_ocean_grid": "SIA",
    "sea_ice_mass_per_area": "SIM",
    "sea_ice_thickness": "SIT",
    "surface_air_pressure": "SAP",
    "sea_level_pressure": "SLP",
    "sea_surface_salinity": "SSS",
}


def download(client, experiment, model, variable, years, out_nc: Path) -> None:
    request = {
        "temporal_resolution": "monthly",
        "experiment": experiment,
        "variable": variable,
        "model": model,
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "format": "zip",
    }
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "download.zip"
        client.retrieve(DATASET, request).download(str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not names:
                raise FileNotFoundError(f"No .nc file in the download for {variable}")
            extracted = zf.extract(names[0], tmp)
        shutil.move(extracted, out_nc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="historical", help="historical, ssp1_2_6, ssp2_4_5, ssp5_8_5, ...")
    p.add_argument("--start", type=int, default=1926)
    p.add_argument("--end", type=int, default=2014)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--variables", nargs="+", default=list(VARIABLES))
    p.add_argument("--out", default="data/raw/cmip6")
    a = p.parse_args()

    import cdsapi  # imported here so --help works without it installed

    client = cdsapi.Client()
    years = range(a.start, a.end + 1)
    failures = []
    for model in a.models:
        model_dir = Path(a.out) / a.experiment / model
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {a.experiment} / {model}")
        for variable in a.variables:
            out_nc = model_dir / f"{variable}.nc"
            if out_nc.exists():
                print(f"  skip {variable} (already downloaded)")
                continue
            print(f"  downloading {variable} ...")
            try:
                download(client, a.experiment, model, variable, years, out_nc)
                print(f"  saved {out_nc}")
            except Exception as exc:  # keep going; report at the end
                failures.append((model, variable, str(exc)))
                print(f"  FAILED {variable}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; rerun to retry them.")


if __name__ == "__main__":
    main()
