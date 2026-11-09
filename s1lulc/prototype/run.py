"""Prototype orchestration: lat/lon + start month -> 12 maps, statistics and a report.

    validate input -> AOI grid -> fetch 12 monthly Sentinel-1 composites (shared recipe)
    -> tiled model inference -> analysis -> GeoTIFF / CSV / JSON / figures / HTML

``run_prototype`` fetches from Earth Engine; ``process`` does everything after the fetch and
takes arrays, so the whole downstream pipeline is testable offline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..geo import PixelGrid
from ..inference.bundle import load_bundle
from ..inference.predictor import predict_series
from . import outputs as O
from .analysis import DEFAULT_EFFECTIVE_N, analyse
from .report import build_html, data_uri
from .window import Month, NoDataError, analysis_window, validate_location

log = logging.getLogger(__name__)


def wgs84_bounds(grid: PixelGrid) -> tuple[float, float, float, float]:
    from rasterio.warp import transform_bounds

    return transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)


def process(s1: dict, grid: PixelGrid, months: list[Month], bundle_dir: str | Path,
            out_dir: str | Path, cfg: Config, lat: float, lon: float,
            dw: dict | None = None, device: str | None = None, smoothing_window: int = 3,
            effective_n: float = DEFAULT_EFFECTIVE_N, tile: int = 256,
            overlap: int = 64) -> dict:
    import torch

    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_bundle(bundle_dir, device, expected_recipe_version=cfg.recipe.version)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [m.label for m in months]
    ignore, colors = meta.ignore_index, meta.class_colors

    log.info("running model on %d months (%s)", len(months), device)
    probs, valid = predict_series(model, s1["db"], meta.norm_stats, meta.n_classes,
                                  device=device, tile=tile, overlap=overlap,
                                  temporal_context=meta.temporal_context)
    dw_mapped = cfg.classes.map_dynamicworld(dw["labels"]) if dw else None
    try:
        res = analyse(probs.astype(np.float32), valid, meta.class_names, ignore,
                      confusion_val=meta.confusion_val or None,
                      noise_floor=meta.noise_floor or None,
                      passes=s1.get("passes"), dw_labels=dw_mapped,
                      smoothing_window=smoothing_window, effective_n=effective_n,
                      platforms=s1.get("platforms"), scenes=s1.get("scenes"))
    except ValueError as e:              # e.g. no pixel valid in all months
        raise NoDataError(str(e)) from e

    # GeoTIFFs
    O.write_geotiff(out / "lulc_monthly.tif", res["maps"]["raw"], grid, ignore, labels)
    O.write_geotiff(out / "lulc_monthly_smoothed.tif", res["maps"]["smoothed"], grid, ignore,
                    labels)
    O.write_geotiff(out / "uncertainty_mean.tif", res["maps"]["entropy_mean"], grid, np.nan)
    O.write_geotiff(out / "label_switches.tif", res["maps"]["raw_change_frequency"], grid, -1)
    if dw_mapped is not None:
        O.write_geotiff(out / "dynamic_world_monthly.tif", dw_mapped, grid, ignore, labels)
    O.write_tables(out, res, labels)
    if res.get("platforms") is not None:
        O.write_acquisitions(out / "acquisitions.csv", res, labels,
                             dw["images"] if dw else None)

    # figures
    import matplotlib.pyplot as plt

    figs = {
        "Monthly maps (raw)": O.figure_map_grid(res["maps"]["raw"], colors, meta.class_names,
                                                labels, "Sentinel-1 land cover, monthly"),
        "Class shares": O.figure_shares(res, colors, labels),
        "Month-to-month change": O.figure_changes(res, labels),
        "Uncertainty, label switches, gross vs net": O.figure_uncertainty(res, colors),
    }
    if dw_mapped is not None:
        figs["Monthly maps (Dynamic World)"] = O.figure_map_grid(
            dw_mapped, colors, meta.class_names, labels, "Dynamic World (independent), monthly")
        figs["Dynamic World agreement"] = O.figure_dw(res, labels)
    charts = {}
    for name, fig in figs.items():
        png = O.fig_to_png(fig)
        fname = name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        (out / f"{fname}.png").write_bytes(png)
        charts[name] = data_uri(png)
        plt.close(fig)

    # interactive layers (downsampled PNGs embedded in the HTML)
    layers = {
        "Model (raw)": [data_uri(O.png_bytes(O.label_rgba(m, colors, ignore)))
                        for m in res["maps"]["raw"]],
        "Model (smoothed)": [data_uri(O.png_bytes(O.label_rgba(m, colors, ignore)))
                             for m in res["maps"]["smoothed"]],
        "Radar VV (dB)": [data_uri(O.png_bytes(O.gray_rgba(d[0]))) for d in s1["db"]],
    }
    if dw_mapped is not None:
        layers["Dynamic World"] = [data_uri(O.png_bytes(O.label_rgba(m, colors, ignore)))
                                   for m in dw_mapped]

    nf = meta.noise_floor.get("raw", {}) if meta.noise_floor else {}
    ctx = {"lat": lat, "lon": lon, "months": labels, "variant": meta.variant,
           "class_colors": colors, "area_km2": grid.width * grid.height * grid.pixel_size ** 2
           / 1e6, "bounds_wgs84": wgs84_bounds(grid), "effective_n": effective_n,
           "noise_quantile": nf.get("quantile", 95),
           "dw_images": dw.get("images") if dw else None,
           "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    (out / "report.html").write_text(build_html(res, ctx, layers, charts), encoding="utf-8")

    summary = {"input": {"lat": lat, "lon": lon, "start": labels[0], "end": labels[-1]},
               "grid": {"crs": grid.crs, "transform": grid.transform, "width": grid.width,
                        "height": grid.height},
               "model": {"variant": meta.variant, "recipe_version": meta.recipe_version,
                         "test_summary": meta.test_summary},
               "acquisitions": {"scenes": s1.get("scenes"), "platforms": s1.get("platforms"),
                                "passes_median": res.get("passes_median")},
               "dynamic_world_images": dw.get("images") if dw else None,
               "runtime_seconds": round(time.time() - t0, 1), "results": res}
    O.write_json(out / "results.json", summary)
    log.info("report written to %s", out / "report.html")
    return res


def check_data(s1: dict, months: list[Month], min_months: int = 2,
               min_fraction: float = 0.01) -> None:
    """Fail early with a clear message when Sentinel-1 has (almost) no data for the area.

    Typical causes: open ocean or polar areas (no IW VV+VH acquisitions), or a location at the
    edge of the acquisition footprint.
    """
    valid = np.all(np.isfinite(s1["db"]), axis=1).reshape(len(months), -1).mean(1)
    good = int((valid >= min_fraction).sum())
    if good < min_months:
        raise NoDataError(
            f"Sentinel-1 IW VV+VH data cover this area in only {good} of 12 months "
            f"({months[0].label} .. {months[-1].label}). Typical causes: open sea, polar "
            "regions or areas acquired in another mode/polarisation. Choose a land location.")


def run_prototype(lat: float, lon: float, start: str, bundle_dir: str | Path,
                  out_dir: str | Path, project: str | None = None,
                  cache_dir: str | Path | None = "cache", dynamic_world: bool = True,
                  config_dir: str | Path | None = None, **kw) -> dict:
    import ee

    from ..gee.recipe import aoi_grid
    from .fetch import fetch_dynamic_world, fetch_s1

    validate_location(lat, lon)
    months = analysis_window(start)
    cfg = load_config(config_dir)
    ee.Initialize(project=project)
    grid = aoi_grid(lon, lat, cfg.recipe)
    cache = Path(cache_dir) if cache_dir else None
    log.info("area of interest %s, %dx%d px, %s .. %s", grid.crs, grid.width, grid.height,
             months[0].label, months[-1].label)
    s1 = fetch_s1(grid, months, cfg.recipe, cache)
    check_data(s1, months)
    dw = fetch_dynamic_world(grid, months, cache) if dynamic_world else None
    return process(s1, grid, months, bundle_dir, out_dir, cfg, lat, lon, dw=dw, **kw)
