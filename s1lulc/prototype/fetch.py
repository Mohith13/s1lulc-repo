"""Data retrieval for the prototype: monthly Sentinel-1 composites and Dynamic World.

Sentinel-1 uses the SAME recipe function as the training export (``gee.recipe``). Pixels are
requested synchronously with ``ee.data.computePixels`` on the snapped UTM grid (one request
per month, well below the 48 MB request limit for 10 x 10 km). Production safeguards:
retries with exponential backoff for transient errors and quota limits, and an on-disk cache
so re-running a location never re-downloads data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..config import Recipe
from ..geo import PixelGrid
from ..preprocessing import linear_stack_to_db
from .window import Month

log = logging.getLogger(__name__)
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_NODATA = 255


def ee_grid(grid: PixelGrid) -> dict:
    return {"dimensions": {"width": grid.width, "height": grid.height},
            "affineTransform": {"scaleX": grid.pixel_size, "shearX": 0, "translateX": grid.x0,
                                "shearY": 0, "scaleY": -grid.pixel_size,
                                "translateY": grid.y1},
            "crsCode": grid.crs}


def compute_pixels(image, grid: PixelGrid, retries: int = 5, wait: float = 2.0) -> np.ndarray:
    """Synchronous pixel request with exponential backoff. Returns a structured array."""
    import ee

    for attempt in range(retries):
        try:
            return ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                                          "grid": ee_grid(grid)})
        except ee.EEException as e:
            msg = str(e).lower()
            transient = any(k in msg for k in ("quota", "rate", "429", "timeout", "deadline",
                                               "unavailable", "internal", "too many"))
            if not transient or attempt == retries - 1:
                raise
            delay = wait * 2 ** attempt
            log.warning("Earth Engine busy (%s); retry in %.0fs", e, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _cache_path(cache_dir: Path | None, kind: str, grid: PixelGrid, m: Month,
                version: str) -> Path | None:
    if cache_dir is None:
        return None
    key = f"{kind}_{grid.epsg}_{int(grid.x0)}_{int(grid.y1)}_{grid.width}_{m.label}_r{version}"
    return Path(cache_dir) / f"{key}.npz"


def fetch_s1(grid: PixelGrid, months: list[Month], recipe: Recipe,
             cache_dir: Path | None = None) -> dict:
    """-> {'db': (12, 2, H, W) float32 NaN=no data, 'passes': (12, H, W) uint8,
           'platforms': list[list[str]], 'scenes': list[int]}"""
    import ee

    from ..gee.recipe import grid_geometry, monthly_composite, s1_collection

    region = grid_geometry(grid)
    pols = recipe.polarisations
    db, passes, platforms, scenes = [], [], [], []
    for m in months:
        cp = _cache_path(cache_dir, "s1", grid, m, recipe.version)
        if cp and cp.exists():
            with np.load(cp, allow_pickle=False) as z:
                d, n, plat, sc = z["db"], z["passes"], list(z["platforms"]), int(z["scenes"])
        else:
            log.info("fetching Sentinel-1 %s", m.label)
            img = monthly_composite(region, m.year, m.month, recipe, suffix="x").unmask(0)
            arr = compute_pixels(img, grid)
            lin = np.stack([arr[f"{p}_x"].astype(np.float32) for p in pols])
            d = linear_stack_to_db(lin)
            n = np.clip(arr["n_x"], 0, 255).astype(np.uint8)
            start = ee.Date.fromYMD(m.year, m.month, 1)
            col = s1_collection(region, start, start.advance(1, "month"), recipe)
            info = ee.Dictionary({"p": col.aggregate_array("platform_number").distinct(),
                                  "n": col.size()}).getInfo()
            plat, sc = sorted(str(p) for p in info["p"]), int(info["n"])
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, db=d.astype(np.float16), passes=n,
                                    platforms=np.array(plat, dtype=str), scenes=sc)
        db.append(np.asarray(d, np.float32))
        passes.append(n)
        platforms.append(plat)
        scenes.append(sc)
    return {"db": np.stack(db), "passes": np.stack(passes), "platforms": platforms,
            "scenes": scenes}


def fetch_dynamic_world(grid: PixelGrid, months: list[Month], cache_dir: Path | None = None,
                        version: str = "dw1") -> dict:
    """Monthly mode of Dynamic World labels. -> {'labels': (12, H, W) uint8 (255 = none),
    'images': list[int]}. Dynamic World is independent of our model (Sentinel-2 based)."""
    import ee

    from ..gee.recipe import grid_geometry

    region = grid_geometry(grid)
    labels, images = [], []
    for m in months:
        cp = _cache_path(cache_dir, "dw", grid, m, version)
        if cp and cp.exists():
            with np.load(cp) as z:
                lab, n = z["labels"], int(z["images"])
        else:
            log.info("fetching Dynamic World %s", m.label)
            start = ee.Date.fromYMD(m.year, m.month, 1)
            col = (ee.ImageCollection(DW_COLLECTION).filterBounds(region)
                   .filterDate(start, start.advance(1, "month")).select("label"))
            n = int(col.size().getInfo())
            if n == 0:
                lab = np.full((grid.height, grid.width), DW_NODATA, np.uint8)
            else:
                img = col.mode().rename("label").unmask(DW_NODATA).toUint8()
                lab = compute_pixels(img, grid)["label"].astype(np.uint8)
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, labels=lab, images=n)
        labels.append(lab)
        images.append(n)
    return {"labels": np.stack(labels), "images": images}
