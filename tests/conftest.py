import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path):
    """A small fake KRILLBASE file with the real column layout, so tests run without data."""
    rng = np.random.default_rng(0)
    n = 800
    years = np.sort(rng.uniform(1930, 2016, n))
    dates = pd.to_datetime([f"{int(y)}-{rng.integers(1, 13):02d}-15" for y in years])
    frac = dates.year + (dates.dayofyear - 1) / 365.25
    lat = rng.uniform(-70, -50, n)
    lon = rng.choice([-179.5, -178.0, 178.0, 179.5, -60.0, -45.0, 20.0, 100.0], n) + rng.normal(0, 0.3, n)
    krill = np.where(rng.random(n) < 0.36, 0.0, np.round(rng.lognormal(2, 1.5, n)))
    sia = np.where(rng.random(n) < 0.8, 0.0, rng.uniform(1, 90, n))
    sit = np.where(sia > 0, rng.uniform(0.1, 2.0, n), np.nan)
    frame = pd.DataFrame({
        "DATE": dates.strftime("%Y-%m-%d"), "FRACTIONAL_YEAR": frac, "LATITUDE": lat, "LONGITUDE": lon,
        "NUMBER_OF_KRILL_UNDER_1M2": krill, "SST": rng.normal(1, 1.5, n), "SIA": sia, "SIM": sia * 5,
        "SIT": sit, "SSS": rng.normal(34, 0.3, n), "SAP": rng.normal(99000, 500, n), "SLP": rng.normal(99000, 500, n),
    })
    frame.loc[rng.choice(n, 10, replace=False), ["SST", "SIA", "SIM", "SSS"]] = np.nan
    path = tmp_path / "krill.csv"
    frame.to_csv(path, index=False)
    return path
