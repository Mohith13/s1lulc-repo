"""End-to-end pipeline test on synthetic GeoTIFFs: build -> train -> evaluate -> load bundle.

Skipped automatically when the heavy optional dependencies are not installed.
"""

import json

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from s1lulc.config import load_config  # noqa: E402
from s1lulc.data.build import build_dataset, class_weights, region_name_from_file  # noqa: E402
from s1lulc.data.dataset import RandomPatchDataset, load_regions  # noqa: E402
from s1lulc.evaluate import evaluate_bundle  # noqa: E402
from s1lulc.inference.bundle import BundleMismatchError, load_bundle  # noqa: E402
from s1lulc.inference.predictor import predict_probs, tile_positions  # noqa: E402
from s1lulc.preprocessing import NormStats  # noqa: E402
from s1lulc.train import load_train_config, lr_lambda, train_variant  # noqa: E402
from tests.synthetic import make_workspace, tiny_train_config  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    cfg_dir, tif_dir = make_workspace(root)
    meta = build_dataset(tif_dir, root / "dataset", load_config(cfg_dir))
    return {"root": root, "cfg_dir": cfg_dir, "data": root / "dataset" / "data", "meta": meta}


def test_region_name_from_file():
    from pathlib import Path
    assert region_name_from_file(Path("paris__r1-0.tif")) == "paris"
    assert region_name_from_file(Path("paris (1).tif")) == "paris"
    assert region_name_from_file(Path("pilot_urban_milan.tif")) == "pilot_urban_milan"


def test_class_weights():
    w = class_weights(np.array([900, 100, 0]))
    assert w[2] == 0 and w[1] > w[0]
    assert np.mean([x for x in w if x > 0]) == pytest.approx(1, abs=1e-3)


def test_build_outputs(workspace):
    data, meta = workspace["data"], workspace["meta"]
    assert len(meta["regions"]) == 4
    assert all(q["ok"] for q in meta["qc"])
    with np.load(data / "syn_a.npz") as z:
        assert z["db"].shape == (12, 2, 128, 128) and z["db"].dtype == np.float16
        assert z["label"].max() < 7 and z["count"].shape == (12, 128, 128)
        assert np.isnan(z["db"][1, :, :, 0]).all()          # February strip is no-data
    stats = NormStats.load(data / "norm.json")
    assert stats.bands == ["VV", "VH"] and stats.mean[0] > stats.mean[1]
    reports = workspace["root"] / "dataset" / "reports"
    assert (reports / "class_distribution_by_split.csv").exists()
    assert (reports / "passes_per_month.csv").read_text().count("\n") == 5
    qc_warnings = " ".join(w for q in meta["qc"] for w in q["warnings"])
    assert "fewer than 3 passes" in qc_warnings


def test_patch_dataset(workspace):
    data, meta = workspace["data"], workspace["meta"]
    regions = load_regions(data, ("train",))
    stats = NormStats.load(data / "norm.json")
    ds = RandomPatchDataset(regions, stats, 7, 255, patch_size=64, samples_per_epoch=10)
    x, y = ds.sample(3)
    assert x.shape == (2, 64, 64) and y.shape == (64, 64) and x.dtype == np.float32
    x2, _ = ds.sample(3)
    np.testing.assert_array_equal(x, x2)                    # reproducible
    ds.set_epoch(1)
    assert not np.array_equal(ds.sample(3)[0], x)          # new patches each epoch
    assert set(np.unique(y)) <= set(range(7)) | {255}
    assert meta["class_weights"][6] > 0


def test_tile_positions():
    assert tile_positions(1000, 256, 64) == [0, 192, 384, 576, 744]
    assert tile_positions(200, 256, 64) == [0]


def test_lr_schedule():
    assert lr_lambda(0, 100, 10) == pytest.approx(0.1)
    assert lr_lambda(10, 100, 10) == pytest.approx(1.0)
    assert lr_lambda(100, 100, 10) == pytest.approx(0.01)


def test_train_evaluate_and_bundle(workspace):
    root, data = workspace["root"], workspace["data"]
    cfg = tiny_train_config(load_train_config())
    bundle = train_variant(data, root / "train", "tiny", cfg, device="cpu")
    assert (bundle / "model.pt").exists()
    hist = json.loads((root / "train" / "tiny" / "history.json").read_text())
    assert len(hist) == 2 and {"train_loss", "val_miou", "val_nll"} <= set(hist[0])

    # resume: running again with more epochs continues instead of restarting
    cfg["epochs"] = 3
    train_variant(data, root / "train", "tiny", cfg, device="cpu")
    hist = json.loads((root / "train" / "tiny" / "history.json").read_text())
    assert [h["epoch"] for h in hist] == [1, 2, 3]

    res = evaluate_bundle(bundle, data, root / "eval", device="cpu", tile=64, overlap=16)
    assert {"val", "test"} <= set(res["splits"])
    assert len(res["splits"]["test"]["miou_per_month"]) == 12
    assert set(res["noise_floor"]) == {"raw", "smoothed"}
    assert (root / "eval" / "confusion_test.png").exists()

    model, meta = load_bundle(bundle, expected_recipe_version="1.0")
    assert meta.confusion_val and meta.noise_floor
    x = np.zeros((2, 100, 90), np.float32)
    p = predict_probs(model, x, meta.n_classes, tile=64, overlap=16)
    assert p.shape == (7, 100, 90)
    np.testing.assert_allclose(p.sum(0), 1, atol=1e-4)
    with pytest.raises(BundleMismatchError):
        load_bundle(bundle, expected_recipe_version="2.0")


def test_ssl4eo_loader_key_mapping():
    import torch

    from s1lulc.models import build_model, load_ssl4eo_encoder

    model = build_model("unet", "resnet50", 2, 7, weights=None)
    fake = {k: torch.randn_like(v) if v.is_floating_point() else v
            for k, v in model.encoder.state_dict().items()}
    fake["fc.weight"] = torch.zeros(1000, 2048)
    n = load_ssl4eo_encoder(model, fake)
    assert n >= 300
    torch.testing.assert_close(model.encoder.conv1.weight, fake["conv1.weight"])
    with pytest.raises(RuntimeError):
        load_ssl4eo_encoder(model, {"conv1.weight": fake["conv1.weight"]})
