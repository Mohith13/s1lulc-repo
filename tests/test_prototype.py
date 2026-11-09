"""Prototype tests: window validation, analysis maths, and an offline end-to-end run with
synthetic Sentinel-1 and Dynamic World data (Earth Engine replaced by arrays)."""

from datetime import date

import numpy as np
import pytest

from s1lulc.prototype.analysis import analyse
from s1lulc.prototype.window import InputError, analysis_window, validate_location

NAMES = ["a", "b", "c"]


def test_window_valid_and_labels():
    months = analysis_window("2025-10", today=date(2026, 11, 7))
    assert len(months) == 12 and months[0].label == "2025-10" and months[-1].label == "2026-09"
    assert analysis_window("2025-11", today=date(2026, 11, 7))[-1].label == "2026-10"


@pytest.mark.parametrize("start,msg", [("2026-01", "latest possible start month is 2025-11"),
                                       ("2014-06", "2015-01 or later"),
                                       ("10/2025", "look like"), ("2025-13", "1-12")])
def test_window_rejects(start, msg):
    with pytest.raises(InputError, match=msg):
        analysis_window(start, today=date(2026, 11, 7))


def test_location_validation():
    validate_location(48.14, 11.42)
    with pytest.raises(InputError):
        validate_location(80, 10)
    with pytest.raises(InputError):
        validate_location(10, 200)


def _probs(labels: np.ndarray, conf: float = 0.9) -> np.ndarray:
    t, h, w = labels.shape
    p = np.full((t, 3, h, w), (1 - conf) / 2, np.float32)
    for k in range(3):
        p[:, k][labels == k] = conf
    return p


def test_analyse_shares_changes_and_flags():
    lab = np.zeros((12, 4, 5), np.uint8)       # 20 pixels, class 0 everywhere
    lab[6:, :, :2] = 1                         # from July, 8 pixels become class 1
    valid = np.ones_like(lab, bool)
    valid[3, 0, 0] = False                     # pixel missing in April -> excluded,
                                               # so 7 of the 19 common pixels change
    floor = {"raw": {"month_step_pp": [5, 5, 5], "total_pp": [5, 5, 5]},
             "smoothed": {"month_step_pp": [5, 5, 5], "total_pp": [5, 5, 5]}}
    cm_val = np.diag([100, 100, 100])
    res = analyse(_probs(lab), valid, NAMES, 255, confusion_val=cm_val, noise_floor=floor,
                  passes=np.full((12, 4, 5), 9, np.uint8), smoothing_window=1)
    raw = res["series"]["raw"]
    assert res["common_valid_fraction"] == pytest.approx(19 / 20)
    np.testing.assert_allclose(raw["shares"][0], [100, 0, 0])
    np.testing.assert_allclose(raw["shares"][-1], [100 * 12 / 19, 100 * 7 / 19, 0])
    np.testing.assert_allclose(raw["total"], [-700 / 19, 700 / 19, 0])
    assert raw["significant_total"].tolist() == [True, True, False]
    jump = raw["month_to_month"][5]            # June -> July
    np.testing.assert_allclose(jump, [-700 / 19, 700 / 19, 0])
    assert raw["significant_m2m"][5].tolist() == [True, True, False]
    assert not raw["significant_m2m"][0].any()
    np.testing.assert_allclose(raw["adjusted"], raw["shares"])  # perfect confusion -> unchanged
    assert raw["changed_total"] == pytest.approx(700 / 19)
    assert res["warnings"] == []


def test_analyse_smoothing_removes_single_month_flip_and_reports_missing_month():
    lab = np.zeros((12, 3, 3), np.uint8)
    lab[4] = 1                                  # May flips everything to class 1
    p = _probs(lab, conf=0.55)
    valid = np.ones_like(lab, bool)
    valid[9] = False                            # October completely missing
    res = analyse(p, valid, NAMES, 255, smoothing_window=3,
                  passes=np.where(np.arange(12)[:, None, None] == 2, 2, 9).astype(np.uint8)
                  * np.ones((12, 3, 3), np.uint8))
    assert res["available"][9] is False
    assert np.isnan(res["series"]["raw"]["shares"][9]).all()
    assert np.isnan(res["series"]["raw"]["month_to_month"][8]).all()
    assert res["series"]["raw"]["shares"][4, 1] == pytest.approx(100)
    assert res["series"]["smoothed"]["shares"][4, 1] == pytest.approx(0)
    assert res["series"]["smoothed"]["flicker_mean"] < res["series"]["raw"]["flicker_mean"]
    kinds = {w["kind"] for w in res["warnings"]}
    assert {"missing_month", "few_acquisitions"} <= kinds


def test_analyse_dynamic_world_agreement():
    lab = np.zeros((12, 4, 4), np.uint8)
    dw = lab.copy()
    dw[:, 0, :] = 2                             # DW disagrees on one row (25%)
    dw[0] = 255                                 # January: no cloud-free DW image
    res = analyse(_probs(lab), np.ones_like(lab, bool), NAMES, 255, dw_labels=dw,
                  smoothing_window=1)
    d = res["dynamic_world"]
    assert d["agreement"][0] is None and d["agreement"][1] == pytest.approx(0.75)
    assert d["mean_agreement"] == pytest.approx(0.75)


def test_analyse_rejects_no_data():
    with pytest.raises(ValueError):
        analyse(np.zeros((12, 3, 2, 2), np.float32), np.zeros((12, 2, 2), bool), NAMES, 255)


# --- offline end-to-end ----------------------------------------------------------------------
pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("segmentation_models_pytorch")


def test_process_end_to_end(tmp_path):
    import json

    from s1lulc.config import load_config
    from s1lulc.data.build import build_dataset
    from s1lulc.geo import snap_grid
    from s1lulc.prototype.run import process
    from s1lulc.prototype.window import analysis_window
    from s1lulc.train import load_train_config, train_variant
    from tests.synthetic import make_label, make_stack, make_workspace, tiny_train_config

    cfg_dir, tifs = make_workspace(tmp_path, size=96)
    build_dataset(tifs, tmp_path / "ds", load_config(cfg_dir))
    tcfg = tiny_train_config(load_train_config())
    tcfg["epochs"] = 1
    bundle = train_variant(tmp_path / "ds" / "data", tmp_path / "tr", "tiny", tcfg, device="cpu")

    rng = np.random.default_rng(5)
    stack = make_stack(make_label(96, rng), rng)
    lin = stack[:36].reshape(12, 3, 96, 96)
    from s1lulc.preprocessing import linear_stack_to_db
    s1 = {"db": linear_stack_to_db(lin[:, :2]), "passes": lin[:, 2].astype(np.uint8),
          "scenes": [9] * 12, "platforms": [["A", "B"]] * 12}
    dw = {"labels": np.full((12, 96, 96), 1, np.uint8), "images": [3] * 12}
    grid = snap_grid(691_000, 5_335_000, 32632, half_size=480, pixel_size=10)
    months = analysis_window("2025-10", today=date(2026, 11, 7))
    out = tmp_path / "pred"
    res = process(s1, grid, months, bundle, out, load_config(cfg_dir), 48.14, 11.42, dw=dw,
                  device="cpu", tile=64, overlap=16)
    for f in ["report.html", "results.json", "class_shares.csv", "change_month_to_month.csv",
              "change_total.csv", "transitions_first_to_last.csv", "lulc_monthly.tif",
              "uncertainty_mean.tif", "dynamic_world_monthly.tif",
              "dynamic_world_agreement.csv"]:
        assert (out / f).exists(), f
    html = (out / "report.html").read_text()
    assert "Land cover change report" in html and html.count("data:image/png;base64") > 40
    summary = json.loads((out / "results.json").read_text())
    assert summary["input"]["start"] == "2025-10"
    np.testing.assert_allclose(np.nansum(res["series"]["raw"]["shares"], 1)[res["available"]],
                               100, atol=1e-6)
    import rasterio
    with rasterio.open(out / "lulc_monthly.tif") as src:
        assert src.count == 12 and src.crs.to_epsg() == 32632 and src.descriptions[0] == "2025-10"


def test_sensor_change_detection_and_warning():
    from s1lulc.prototype.analysis import sensor_changes

    plat = [["A", "C"]] * 8 + [["A", "D"], ["C", "D"], ["C", "D"], ["C", "D"]]
    avail = [True] * 12
    assert sensor_changes(plat, avail) == [8, 9]
    avail[8] = False
    assert sensor_changes(plat, avail) == [9]
    lab = np.zeros((12, 2, 2), np.uint8)
    res = analyse(_probs(lab), np.ones_like(lab, bool), NAMES, 255, smoothing_window=1,
                  platforms=plat, scenes=[10] * 12)
    w = [x for x in res["warnings"] if x["kind"] == "sensor_change"]
    assert w and w[0]["months"] == [8, 9]


def test_check_data_rejects_area_without_radar_data():
    from datetime import date

    from s1lulc.prototype.run import check_data
    from s1lulc.prototype.window import NoDataError, analysis_window

    months = analysis_window("2025-10", today=date(2026, 11, 7))
    db = np.full((12, 2, 4, 4), np.nan, np.float32)
    with pytest.raises(NoDataError, match="only 0 of 12 months"):
        check_data({"db": db}, months)
    db[3:5] = -12.0
    check_data({"db": db}, months)            # two months with data is enough


def test_cli_reports_input_errors_cleanly(capsys):
    from s1lulc.cli import main

    code = main(["predict", "--lat", "48.1", "--lon", "11.4", "--start", "2030-01",
                 "--bundle", "x", "--project", "p"])
    assert code == 2
    assert "latest possible start month" in capsys.readouterr().err
