"""Dataset export: one GeoTIFF per region to Google Drive, plus a manifest.

Production safeguards:
* idempotent - regions with a READY/RUNNING/COMPLETED task of the same name are skipped
  (re-running the command never creates duplicate files),
* every export uses an explicit CRS + affine transform (identical radar/label grid),
* a manifest records grid, recipe version and task id for each region (reproducibility).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config, Region
from .recipe import aoi_grid, band_names, grid_geometry, training_stack

log = logging.getLogger(__name__)

ACTIVE_OR_DONE = {"READY", "RUNNING", "COMPLETED", "PENDING"}


def task_name(region: Region, recipe_version: str) -> str:
    """Export description; includes the recipe version so a new recipe never clashes."""
    return f"{region.name}__r{recipe_version.replace('.', '-')}"


def regions_to_skip(existing_tasks: Iterable[dict], wanted: Iterable[str]) -> set[str]:
    """Names in ``wanted`` that already have an active or completed export task.

    ``existing_tasks`` are task status dicts as returned by ``ee.data.getTaskList()``
    (keys ``description`` and ``state``). Pure function, so it is unit-tested.
    """
    wanted = set(wanted)
    return {t.get("description") for t in existing_tasks
            if t.get("description") in wanted and t.get("state") in ACTIVE_OR_DONE}


def export_regions(cfg: Config, folder: str, regions: list[Region] | None = None,
                   manifest_path: Path | None = None, dry_run: bool = False,
                   force: bool = False) -> list[dict]:
    import ee

    regions = regions if regions is not None else cfg.regions
    names = {task_name(r, cfg.recipe.version): r for r in regions}
    skip = set() if force else regions_to_skip(ee.data.getTaskList(), names)

    records = []
    for name, region in names.items():
        if name in skip:
            log.info("skip %s (task already exists)", name)
            continue
        grid = aoi_grid(region.lon, region.lat, cfg.recipe)
        rec = {
            "region": region.name, "split": region.split, "lon": region.lon, "lat": region.lat,
            "file": f"{name}.tif", "crs": grid.crs, "transform": grid.transform,
            "width": grid.width, "height": grid.height,
            "bands": band_names(cfg.recipe), "recipe_version": cfg.recipe.version,
            "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if not dry_run:
            task = ee.batch.Export.image.toDrive(
                image=training_stack(grid, cfg.recipe), description=name, folder=folder,
                fileNamePrefix=name, region=grid_geometry(grid), crs=grid.crs,
                crsTransform=grid.transform, maxPixels=1e9, fileFormat="GeoTIFF")
            task.start()
            rec["task_id"] = task.id
        log.info("%s %s (%s, %dx%d)", "planned" if dry_run else "started", name, grid.crs,
                 grid.width, grid.height)
        records.append(rec)

    if manifest_path and records:
        manifest_path = Path(manifest_path)
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        merged = {r["region"]: r for r in old} | {r["region"]: r for r in records}
        manifest_path.write_text(json.dumps(list(merged.values()), indent=2))
    return records
