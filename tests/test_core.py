"""Tests for configuration, geometry, preprocessing and export bookkeeping."""

import numpy as np
import pytest

from s1lulc.config import load_config
from s1lulc.gee.export import regions_to_skip, task_name
from s1lulc.gee.recipe import band_names
from s1lulc.geo import snap_grid, utm_epsg
from s1lulc.preprocessing import NormStats, linear_to_db, normalise, valid_mask


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_configs_load_and_are_consistent(cfg):
    assert cfg.classes.n_classes == 7
    assert len(cfg.regions) == len({r.name for r in cfg.regions}), "duplicate region names"
    splits = {r.split for r in cfg.regions}
    assert {"train", "val", "test"} <= splits


def test_worldcover_mapping(cfg):
    codes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0, np.nan])
    got = cfg.classes.map_worldcover(codes)
    assert got.tolist() == [0, 1, 1, 2, 3, 4, 255, 5, 6, 6, 1, 255, 255]


def test_dynamicworld_mapping(cfg):
    got = cfg.classes.map_dynamicworld(np.arange(9))
    assert got.tolist() == [5, 0, 1, 6, 2, 1, 3, 4, 255]


@pytest.mark.parametrize("lon,lat,epsg", [
    (9.19, 45.46, 32632),     # Milan, zone 32N
    (11.12, 48.00, 32632),    # Ammersee
    (-46.63, -23.55, 32723),  # Sao Paulo, zone 23S
    (31.24, 30.04, 32636),    # Cairo
    (179.9, 10.0, 32660),
])
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


def test_snap_grid_is_aligned_and_square():
    g = snap_grid(514_327.4, 5_034_881.9, 32632, half_size=5000, pixel_size=10)
    assert (g.width, g.height) == (1000, 1000)
    assert g.x0 % 10 == 0 and g.y1 % 10 == 0
    xmin, ymin, xmax, ymax = g.bounds
    assert xmax - xmin == 10_000 and ymax - ymin == 10_000
    assert g.transform == [10.0, 0.0, g.x0, 0.0, -10.0, g.y1]
    # the requested centre lies within one pixel of the box centre
    assert abs((xmin + xmax) / 2 - 514_327.4) <= 10 and abs((ymin + ymax) / 2 - 5_034_881.9) <= 10


def test_band_names_match_pilot_layout(cfg):
    names = band_names(cfg.recipe)
    assert len(names) == 37
    assert names[:3] == ["VV_01", "VH_01", "n_01"] and names[-1] == "label"


def test_linear_to_db():
    db = linear_to_db(np.array([1.0, 0.1, 0.01, 0.0, np.nan]))
    np.testing.assert_allclose(db[:3], [0.0, -10.0, -20.0], atol=1e-5)
    assert db[3] == pytest.approx(-60.0)       # floored, not -inf
    assert np.isnan(db[4])


def test_normalise_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    lin = rng.uniform(0.002, 0.5, size=(2, 64, 64))
    lin[:, 0, 0] = np.nan                       # one invalid pixel
    db = linear_to_db(lin).reshape(2, -1)
    stats = NormStats.fit(db, ["VV", "VH"], (1, 99), "1.0")
    stats.save(tmp_path / "norm.json")
    stats2 = NormStats.load(tmp_path / "norm.json")
    x, valid = normalise(lin, stats2)
    assert x.dtype == np.float32 and x.shape == (2, 64, 64)
    assert not valid[0, 0] and valid.sum() == 64 * 64 - 1
    assert np.all(x[:, 0, 0] == 0)
    assert abs(float(x[:, valid].mean())) < 0.05 and abs(float(x[:, valid].std()) - 1) < 0.05


def test_normalise_rejects_wrong_band_count():
    stats = NormStats(["VV", "VH"], [-25, -30], [5, 0], [-10, -17], [4, 4], "1.0")
    with pytest.raises(ValueError):
        normalise(np.ones((3, 4, 4)), stats)


def test_valid_mask():
    s = np.ones((2, 2, 2))
    s[1, 0, 1] = 0.0
    s[0, 1, 1] = np.nan
    assert valid_mask(s).tolist() == [[True, False], [True, False]]


def test_export_is_idempotent():
    tasks = [
        {"description": "paris__r1-0", "state": "COMPLETED"},
        {"description": "madrid__r1-0", "state": "FAILED"},
        {"description": "warsaw__r1-0", "state": "RUNNING"},
        {"description": "other", "state": "READY"},
    ]
    wanted = ["paris__r1-0", "madrid__r1-0", "warsaw__r1-0", "athens__r1-0"]
    assert regions_to_skip(tasks, wanted) == {"paris__r1-0", "warsaw__r1-0"}


def test_task_name_contains_recipe_version(cfg):
    assert task_name(cfg.regions[0], "1.0") == f"{cfg.regions[0].name}__r1-0"
