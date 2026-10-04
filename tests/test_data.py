"""Guards against the data problems fixed in preprocessing v2 (see docs/CHANGES_preprocessing_v2.md)."""
import numpy as np
import pandas as pd

from fishcast.data import PREPROCESSING_VERSION, add_polar_coords, load_krill_dataset


def load(path, **kwargs):
    return load_krill_dataset(path, num_inducing_points=20, **kwargs)


def test_splits_are_chronological_and_disjoint(synthetic_csv):
    b = load(synthetic_csv)
    years = b.dataframe["FRACTIONAL_YEAR"].to_numpy()
    assert len(b.test_indices) > 0
    assert years[b.train_indices].max() < years[b.val_indices].min()
    assert years[b.val_indices].max() < years[b.test_indices].min()
    assert len(set(b.train_indices) & set(b.val_indices) & set(b.test_indices)) == 0


def test_scaling_fitted_on_training_rows_only(synthetic_csv):
    b = load(synthetic_csv)
    sst = b.feature_columns.index("SST")
    train_mean = b.features[b.train_indices, sst].mean().item()
    assert abs(train_mean) < 1e-4


def test_lag_features_line_up_with_targets(synthetic_csv):
    raw = pd.read_csv(synthetic_csv)
    b = load(synthetic_csv, include_lag_features=True)
    assert np.allclose(b.dataframe["LATITUDE"].to_numpy(), raw["LATITUDE"].to_numpy())
    assert np.allclose(np.log1p(raw["NUMBER_OF_KRILL_UNDER_1M2"].to_numpy()), b.target.numpy().ravel(), atol=1e-5)


def test_no_empty_haul_is_a_hotspot(synthetic_csv):
    b = load(synthetic_csv)
    krill = b.dataframe["NUMBER_OF_KRILL_UNDER_1M2"].to_numpy()
    assert not ((b.hotspot.numpy().ravel() == 1) & (krill == 0)).any()


def test_ice_free_hauls_get_zero_thickness(synthetic_csv):
    raw = pd.read_csv(synthetic_csv)
    b = load(synthetic_csv)
    ice_free = (raw["SIA"] == 0).to_numpy()
    sit_raw = b.feature_stats["SIT"]
    sit = b.features[:, b.feature_columns.index("SIT")].numpy() * sit_raw["std"] + sit_raw["mean"]
    assert np.allclose(sit[ice_free], 0.0, atol=1e-4)


def test_missing_flags_added(synthetic_csv):
    b = load(synthetic_csv)
    assert "SST_MISSING" in b.feature_columns


def test_antimeridian_neighbours_are_close():
    pts = add_polar_coords(pd.DataFrame({"LATITUDE": [-60.0, -60.0], "LONGITUDE": [179.9, -179.9]}))
    gap = np.hypot(*(pts[["POLAR_X", "POLAR_Y"]].iloc[0] - pts[["POLAR_X", "POLAR_Y"]].iloc[1]))
    assert gap < 0.2


def test_legacy_pipeline_still_available(synthetic_csv):
    b = load(synthetic_csv, preprocessing_version=1)
    assert b.metadata["preprocessing_version"] == 1
    assert len(b.test_indices) == 0
    assert PREPROCESSING_VERSION >= 2
