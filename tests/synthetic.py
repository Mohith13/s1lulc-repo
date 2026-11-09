"""Synthetic stand-ins for exported region GeoTIFFs (used by the pipeline tests).

Each synthetic region has blob-shaped classes with class-specific VV/VH backscatter, a
seasonal cycle for cropland, speckle-like noise and a pass-count band, written in exactly
the 37-band layout of the Earth Engine export.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from s1lulc.config import CONFIG_DIR
from s1lulc.geo import utm_epsg

# mean (VV, VH) dB per WorldCover code
SIGNATURES = {10: (-7.5, -13.5), 30: (-10.0, -17.0), 40: (-11.0, -18.0), 50: (-3.0, -11.0),
              60: (-15.0, -23.0), 80: (-21.0, -27.0), 90: (-13.0, -21.0)}

REGIONS = {
    "syn_a": {"lon": 9.19, "lat": 45.46, "split": "train"},
    "syn_b": {"lon": 11.12, "lat": 48.00, "split": "train"},
    "syn_c": {"lon": 16.37, "lat": 48.21, "split": "val"},
    "syn_d": {"lon": 14.42, "lat": 50.08, "split": "test"},
}


def make_label(size: int, rng: np.random.Generator) -> np.ndarray:
    codes = np.array(list(SIGNATURES))
    yy, xx = np.mgrid[:size, :size]
    centres = rng.uniform(0, size, (14, 2))
    d = np.stack([(yy - cy) ** 2 + (xx - cx) ** 2 for cy, cx in centres])
    return codes[np.argmin(d, 0) % len(codes)].astype(np.float32)


def make_stack(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = label.shape[0]
    bands = []
    for m in range(12):
        vv = np.zeros_like(label)
        vh = np.zeros_like(label)
        for code, (mv, mh) in SIGNATURES.items():
            seasonal = 3.0 * np.sin(2 * np.pi * m / 12) if code == 40 else 0.0
            vv[label == code] = mv + seasonal
            vh[label == code] = mh + seasonal
        vv += rng.normal(0, 1.0, vv.shape)
        vh += rng.normal(0, 1.0, vh.shape)
        lin_vv, lin_vh = 10 ** (vv / 10), 10 ** (vh / 10)
        if m == 1:                       # a strip with no data in February
            lin_vv[:, : size // 8] = np.nan
            lin_vh[:, : size // 8] = np.nan
        bands += [lin_vv, lin_vh, np.full_like(vv, 2 if m == 1 else 9)]
    return np.stack([*bands, label]).astype(np.float32)


def write_region_tif(path: Path, lon: float, lat: float, size: int, seed: int) -> None:
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    stack = make_stack(make_label(size, rng), rng)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=37,
                       dtype="float32", crs=f"EPSG:{utm_epsg(lon, lat)}",
                       transform=from_origin(500_000, 5_000_000, 10, 10)) as dst:
        dst.write(stack)


def make_workspace(root: Path, size: int = 128) -> tuple[Path, Path]:
    """Create (config_dir, tif_dir) with synthetic regions; returns both paths."""
    cfg_dir, tif_dir = root / "configs", root / "tifs"
    shutil.copytree(CONFIG_DIR, cfg_dir)
    tif_dir.mkdir()
    (cfg_dir / "regions.yaml").write_text(yaml.safe_dump({"regions": REGIONS}))
    recipe = yaml.safe_load((cfg_dir / "recipe.yaml").read_text())
    recipe["aoi_half_size_m"] = size * 5
    (cfg_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    for i, (name, r) in enumerate(REGIONS.items()):
        write_region_tif(tif_dir / f"{name}__r1-0.tif", r["lon"], r["lat"], size, seed=i)
    return cfg_dir, tif_dir


def tiny_train_config(base: dict) -> dict:
    cfg = dict(base)
    cfg.update(patch_size=64, samples_per_epoch=32, batch_size=8, num_workers=0, epochs=2,
               early_stopping_patience=5, val_months=[1, 2])
    cfg["inference"] = {"tile": 64, "overlap": 16}
    cfg["variants"] = {"tiny": {"arch": "unet", "encoder": "resnet18", "weights": None}}
    return cfg
