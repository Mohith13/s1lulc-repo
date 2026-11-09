"""Tests for multi-month (temporal context) input and round-1 bundle compatibility."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from s1lulc.inference.bundle import read_meta
from s1lulc.preprocessing import NormStats, model_input, month_indices

STATS = NormStats(["VV", "VH"], [-25, -30], [5, 0], [-10, -17], [5, 5], "1.0")


def test_month_indices_edges_repeat():
    assert month_indices(0, 12, 1) == [0, 0, 1]
    assert month_indices(5, 12, 1) == [4, 5, 6]
    assert month_indices(11, 12, 1) == [10, 11, 11]
    assert month_indices(3, 12, 0) == [3]


def test_model_input_shapes_order_and_fallback():
    db = np.stack([np.full((2, 4, 4), -10.0 - t, np.float32) for t in range(12)])
    db[4, :, 0, 0] = np.nan                    # neighbour month gap at one pixel
    x, valid = model_input(db, 5, STATS, 1)
    assert x.shape == (6, 4, 4) and valid.all()
    np.testing.assert_allclose(x[0, 1, 1], (-14 + 10) / 5)    # month 4 first (oldest)
    np.testing.assert_allclose(x[2, 1, 1], (-15 + 10) / 5)    # then month 5
    np.testing.assert_allclose(x[4, 1, 1], (-16 + 10) / 5)    # then month 6
    np.testing.assert_allclose(x[0, 0, 0], x[2, 0, 0])        # gap filled from month 5
    x0, _ = model_input(db, 5, STATS, 0)
    np.testing.assert_allclose(x0, x[2:4])                    # k=0 equals the centre slice
    db[5, :, 1, 1] = np.nan                    # centre month gap -> pixel invalid
    _, valid = model_input(db, 5, STATS, 1)
    assert not valid[1, 1] and valid.sum() == 15


def test_round1_bundle_still_loads():
    meta = read_meta_from(Path(__file__).parent / "fixtures" / "round1_bundle.json")
    assert meta.temporal_context == 0 and meta.in_channels == 2
    assert meta.noise_floor and len(meta.confusion_val) == 7


def read_meta_from(path: Path):
    tmp = path.parent / "_bundle_tmp"
    tmp.mkdir(exist_ok=True)
    shutil.copy(path, tmp / "bundle.json")
    try:
        return read_meta(tmp)
    finally:
        shutil.rmtree(tmp)


pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("rasterio")


def test_ssl4eo_conv1_tiled_for_three_months():
    import torch

    from s1lulc.models import build_model, load_ssl4eo_encoder

    m2 = build_model("unet", "resnet50", 2, 7)
    fake = {k: torch.randn_like(v) if v.is_floating_point() else v
            for k, v in m2.encoder.state_dict().items()}
    m6 = build_model("unet", "resnet50", 6, 7)
    load_ssl4eo_encoder(m6, fake)
    w = m6.encoder.conv1.weight
    assert w.shape[1] == 6
    torch.testing.assert_close(w[:, 0:2] * 3, fake["conv1.weight"])
    torch.testing.assert_close(w[:, 4:6], w[:, 0:2])
    # constant series -> same first-layer response as the 2-channel pretrained conv
    x2 = torch.randn(1, 2, 32, 32)
    torch.testing.assert_close(torch.nn.functional.conv2d(x2.repeat(1, 3, 1, 1), w),
                               torch.nn.functional.conv2d(x2, fake["conv1.weight"]),
                               rtol=1e-4, atol=1e-4)


def test_temporal_variant_trains_evaluates_and_predicts(tmp_path):
    from datetime import date

    from s1lulc.config import load_config
    from s1lulc.data.build import build_dataset
    from s1lulc.data.dataset import RandomPatchDataset, load_regions
    from s1lulc.evaluate import evaluate_bundle
    from s1lulc.geo import snap_grid
    from s1lulc.inference.bundle import load_bundle
    from s1lulc.preprocessing import linear_stack_to_db
    from s1lulc.prototype.run import process
    from s1lulc.prototype.window import analysis_window
    from s1lulc.train import load_train_config, train_variant
    from tests.synthetic import make_label, make_stack, make_workspace, tiny_train_config

    cfg_dir, tifs = make_workspace(tmp_path, size=96)
    build_dataset(tifs, tmp_path / "ds", load_config(cfg_dir))
    data = tmp_path / "ds" / "data"
    stats = NormStats.load(data / "norm.json")
    ds = RandomPatchDataset(load_regions(data, ("train",)), stats, 7, 255, patch_size=64,
                            samples_per_epoch=4, temporal_context=1)
    x, y = ds.sample(0)
    assert x.shape == (6, 64, 64) and y.shape == (64, 64)

    cfg = tiny_train_config(load_train_config())
    cfg["epochs"] = 1
    cfg["val_months"] = None
    cfg["variants"] = {"tiny_t3": {"arch": "unet", "encoder": "resnet18", "weights": None,
                                   "temporal_context": 1}}
    bundle = train_variant(data, tmp_path / "tr", "tiny_t3", cfg, device="cpu")
    model, meta = load_bundle(bundle)
    assert meta.temporal_context == 1 and meta.in_channels == 6
    res = evaluate_bundle(bundle, data, tmp_path / "ev", device="cpu", tile=64, overlap=16)
    assert len(res["splits"]["test"]["miou_per_month"]) == 12

    rng = np.random.default_rng(3)
    lin = make_stack(make_label(96, rng), rng)[:36].reshape(12, 3, 96, 96)
    s1 = {"db": linear_stack_to_db(lin[:, :2]), "passes": lin[:, 2].astype(np.uint8)}
    grid = snap_grid(691_000, 5_335_000, 32632, half_size=480, pixel_size=10)
    months = analysis_window("2025-10", today=date(2026, 11, 7))
    out = process(s1, grid, months, bundle, tmp_path / "pred", load_config(cfg_dir), 48.14,
                  11.42, device="cpu", tile=64, overlap=16)
    assert out["maps"]["raw"].shape == (12, 96, 96)
    assert (tmp_path / "pred" / "report.html").exists()
