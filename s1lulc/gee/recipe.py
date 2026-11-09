"""The monthly Sentinel-1 composite recipe, implemented once and shared.

Both the training-data export and the prototype call ``monthly_composite``, so the model
always sees inputs prepared the same way (no training/serving skew).

Earth Engine's COPERNICUS/S1_GRD(_FLOAT) scenes already have thermal-noise removal,
radiometric calibration to sigma0 and terrain correction (SRTM 30 m) applied. This recipe
adds: mode/polarisation/orbit filtering, a residual border-noise mask, and a monthly temporal
mean in linear power (the main speckle filter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Recipe
from ..geo import PixelGrid, snap_grid, utm_epsg

if TYPE_CHECKING:  # pragma: no cover
    import ee


def _ee():
    import ee  # imported lazily so the rest of the package works without earthengine-api

    return ee


def s1_collection(region: ee.Geometry, start: ee.Date, end: ee.Date,
                  recipe: Recipe) -> ee.ImageCollection:
    """Filtered Sentinel-1 scenes for one time window, with the edge mask applied."""
    ee = _ee()
    col = (ee.ImageCollection(recipe.collection)
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", recipe.instrument_mode)))
    for pol in recipe.polarisations:
        col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
    if recipe.orbit_pass != "both":
        col = col.filter(ee.Filter.eq("orbitProperties_pass", recipe.orbit_pass))
    col = col.select(recipe.polarisations)
    thr = recipe.edge_mask_linear
    first = recipe.polarisations[0]
    return col.map(lambda img: img.updateMask(img.select(first).gt(thr)))


def monthly_composite(region: ee.Geometry, year: int, month: int, recipe: Recipe,
                      suffix: str | None = None) -> ee.Image:
    """One month -> bands <pol>_<suffix> (linear mean) + n_<suffix> (scene count per pixel).

    An empty month (no acquisitions, e.g. the Sentinel-1C manoeuvre gap) yields fully masked
    bands instead of failing, so downstream code can detect and report it.
    """
    ee = _ee()
    if not 1 <= month <= 12:
        raise ValueError("month must be 1..12")
    start = ee.Date.fromYMD(year, month, 1)
    col = s1_collection(region, start, start.advance(1, "month"), recipe)
    pols = recipe.polarisations
    empty = ee.Image.constant([0] * len(pols)).rename(pols).updateMask(0)
    reduced = col.median() if recipe.reducer == "median" else col.mean()
    mean = ee.Image(ee.Algorithms.If(col.size().gt(0), reduced, empty))
    count = ee.Image(ee.Algorithms.If(col.size().gt(0), col.select(pols[0]).count(),
                                      ee.Image.constant(0)))
    s = suffix or f"{month:02d}"
    return (mean.rename([f"{p}_{s}" for p in pols]).toFloat()
            .addBands(count.rename(f"n_{s}").toFloat()))


def aoi_grid(lon: float, lat: float, recipe: Recipe) -> PixelGrid:
    """Snapped UTM pixel grid centred on (lon, lat). Calls Earth Engine for the projection."""
    ee = _ee()
    epsg = utm_epsg(lon, lat)
    x, y = ee.Geometry.Point([lon, lat]).transform(f"EPSG:{epsg}", 1).coordinates().getInfo()
    return snap_grid(x, y, epsg, recipe.aoi_half_size_m, recipe.pixel_size_m)


def grid_geometry(grid: PixelGrid) -> ee.Geometry:
    ee = _ee()
    xmin, ymin, xmax, ymax = grid.bounds
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], grid.crs, False)


def training_stack(grid: PixelGrid, recipe: Recipe) -> ee.Image:
    """12 monthly composites of the label year + raw WorldCover codes, as one image.

    Band order: VV_01, VH_01, n_01, ..., VV_12, VH_12, n_12, label  (37 bands for VV+VH).
    Putting labels in the same image guarantees they share the radar pixel grid.
    """
    ee = _ee()
    region = grid_geometry(grid)
    months = [monthly_composite(region, recipe.training_year, m, recipe) for m in range(1, 13)]
    label = ee.ImageCollection(recipe.label_collection).first().select("Map").rename("label")
    return ee.Image.cat([*months, label.toFloat()])


def band_names(recipe: Recipe) -> list[str]:
    """Expected band order of ``training_stack`` (used to validate exported files)."""
    names: list[str] = []
    for m in range(1, 13):
        names += [f"{p}_{m:02d}" for p in recipe.polarisations] + [f"n_{m:02d}"]
    return [*names, "label"]
