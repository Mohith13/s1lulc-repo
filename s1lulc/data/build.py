"""Build the training dataset from the exported region GeoTIFFs.

Input : one 37-band GeoTIFF per region (VV/VH/n for 12 months + raw WorldCover label),
        as written by ``s1lulc export`` or the equivalent notebook cell.
Output: ``<out>/data/``    flat folder, ready to upload as a Kaggle dataset
            <region>.npz   db (12, 2, H, W) float16 [NaN = no data], count (12, H, W) uint8,
                           label (H, W) uint8 target classes, label_raw (H, W) uint8
            meta.json      regions, splits, grids, QC results, class weights, recipe version
            norm.json      normalisation statistics (fitted on TRAINING regions only)
        ``<out>/reports/`` CSV tables and figures for the report

Quality checks per region (failures are recorded and the region is excluded, never silently
used): band count, grid size, CRS vs expected UTM zone, label coverage, months without data.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..geo import utm_epsg
from ..preprocessing import NormStats, linear_stack_to_db

log = logging.getLogger(__name__)

N_MONTHS = 12
EXPECTED_BANDS = 37
SUFFIX = re.compile(r"(__r[0-9A-Za-z-]+)?( \(\d+\))?$")   # "paris__r1-0", "paris (1)"
MIN_LABEL_COVERAGE = 0.5
NORM_SAMPLES_PER_REGION = 200_000


@dataclass
class RegionQC:
    region: str
    split: str
    file: str = ""
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    crs: str = ""
    label_coverage: float = 0.0
    passes_median: list[int] = field(default_factory=list)
    valid_fraction: list[float] = field(default_factory=list)


def region_name_from_file(path: Path) -> str:
    return SUFFIX.sub("", path.stem)


def find_region_files(src: Path, names: list[str]) -> tuple[dict[str, Path], list[str]]:
    """Match GeoTIFFs in ``src`` to region names. Returns (mapping, warnings).

    Drive can contain duplicates ("paris (1).tif") if an export ran twice; the most recent
    file wins and a warning is recorded.
    """
    found: dict[str, list[Path]] = {}
    for p in sorted(Path(src).glob("*.tif")):
        found.setdefault(region_name_from_file(p), []).append(p)
    mapping, warnings = {}, []
    for name in names:
        files = found.get(name, [])
        if not files:
            continue
        files.sort(key=lambda p: p.stat().st_mtime)
        if len(files) > 1:
            warnings.append(f"{name}: {len(files)} files found, using newest {files[-1].name}")
        mapping[name] = files[-1]
    return mapping, warnings


def split_stack(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(37, H, W) export -> linear (12, 2, H, W), count (12, H, W), raw label (H, W)."""
    if data.shape[0] != EXPECTED_BANDS:
        raise ValueError(f"expected {EXPECTED_BANDS} bands, got {data.shape[0]}")
    months = data[:36].reshape(N_MONTHS, 3, *data.shape[1:])
    return months[:, :2], months[:, 2], data[36]


def process_region(path: Path, region, cfg: Config) -> tuple[dict, RegionQC]:
    import rasterio

    qc = RegionQC(region=region.name, split=region.split, file=path.name)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        qc.crs = str(src.crs)
        qc.width, qc.height = src.width, src.height
        transform = list(src.transform)[:6]
        epsg = src.crs.to_epsg() if src.crs else None

    if data.shape[0] != EXPECTED_BANDS:
        qc.ok = False
        qc.errors.append(f"{data.shape[0]} bands, expected {EXPECTED_BANDS}")
        return {}, qc
    expected_px = 2 * cfg.recipe.aoi_half_size_m // cfg.recipe.pixel_size_m
    if abs(qc.width - expected_px) > 2 or abs(qc.height - expected_px) > 2:
        qc.warnings.append(f"size {qc.width}x{qc.height}, expected {expected_px}x{expected_px}")
    if epsg != utm_epsg(region.lon, region.lat):
        qc.warnings.append(f"CRS {qc.crs}, expected EPSG:{utm_epsg(region.lon, region.lat)}")

    linear, count, raw = split_stack(data)
    db = linear_stack_to_db(linear)                                   # NaN where invalid
    raw_u8 = np.where(np.isfinite(raw), raw, 0).clip(0, 255).astype(np.uint8)
    label = cfg.classes.map_worldcover(raw)
    count_u8 = np.nan_to_num(count, nan=0).clip(0, 255).astype(np.uint8)

    qc.label_coverage = float(np.mean(label != cfg.classes.ignore_index))
    month_valid = np.all(np.isfinite(db), axis=1)                     # (12, H, W)
    qc.valid_fraction = [round(float(v.mean()), 4) for v in month_valid]
    qc.passes_median = [int(np.median(c[c > 0])) if np.any(c > 0) else 0 for c in count_u8]
    if qc.label_coverage < MIN_LABEL_COVERAGE:
        qc.ok = False
        qc.errors.append(f"label coverage {qc.label_coverage:.1%} < {MIN_LABEL_COVERAGE:.0%}")
    empty = [m + 1 for m, v in enumerate(qc.valid_fraction) if v < 0.01]
    if len(empty) == N_MONTHS:
        qc.ok = False
        qc.errors.append("no radar data in any month")
    elif empty:
        qc.warnings.append(f"months without radar data: {empty}")
    low = [m + 1 for m, p in enumerate(qc.passes_median) if 0 < p < 3]
    if low:
        qc.warnings.append(f"months with fewer than 3 passes (noisier composite): {low}")

    arrays = {"db": db.astype(np.float16), "count": count_u8, "label": label,
              "label_raw": raw_u8, "transform": np.asarray(transform, np.float64)}
    return arrays, qc


def class_weights(counts: np.ndarray, max_weight: float = 10.0) -> list[float]:
    """Inverse-square-root frequency weights, normalised to mean 1 over present classes.

    Square root (not plain inverse) frequency keeps rare classes from dominating the loss;
    absent classes get weight 0.
    """
    counts = np.asarray(counts, np.float64)
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1e-12)), 0.0)
    present = w > 0
    w[present] /= w[present].mean()
    return np.minimum(w, max_weight).round(4).tolist()


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_dataset(src: Path, out: Path, cfg: Config, seed: int = 42) -> dict:
    src, out = Path(src), Path(out)
    data_dir, report_dir = out / "data", out / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = cfg.classes.names
    n_cls = cfg.classes.n_classes

    files, warnings = find_region_files(src, [r.name for r in cfg.regions])
    for w in warnings:
        log.warning(w)

    qcs: list[RegionQC] = []
    class_counts: dict[str, np.ndarray] = {}
    norm_samples = []
    for region in cfg.regions:
        if region.name not in files:
            qc = RegionQC(region=region.name, split=region.split, ok=False,
                          errors=["file not found"])
            log.warning("%s: file not found in %s", region.name, src)
            qcs.append(qc)
            continue
        log.info("processing %s (%s)", region.name, region.split)
        arrays, qc = process_region(files[region.name], region, cfg)
        qcs.append(qc)
        if not qc.ok:
            log.error("%s excluded: %s", region.name, "; ".join(qc.errors))
            continue
        for w in qc.warnings:
            log.warning("%s: %s", region.name, w)
        np.savez_compressed(data_dir / f"{region.name}.npz", **arrays)
        lab = arrays["label"]
        class_counts[region.name] = np.bincount(lab[lab < n_cls], minlength=n_cls)[:n_cls]
        if region.split == "train":
            db = arrays["db"].astype(np.float32)                       # (12, 2, H, W)
            flat = db.transpose(1, 0, 2, 3).reshape(2, -1)
            ok = np.all(np.isfinite(flat), axis=0).nonzero()[0]
            take = rng.choice(ok, size=min(NORM_SAMPLES_PER_REGION, ok.size), replace=False)
            norm_samples.append(flat[:, take])

    used = [q for q in qcs if q.ok]
    if not any(q.split == "train" for q in used):
        raise RuntimeError("no usable training region; see QC errors above")
    if not any(q.split == "val" for q in used):
        raise RuntimeError("no usable validation region; see QC errors above")

    stats = NormStats.fit(np.concatenate(norm_samples, axis=1), cfg.recipe.polarisations,
                          cfg.recipe.db_clip_percentiles, cfg.recipe.version)
    stats.save(data_dir / "norm.json")

    split_counts = {}
    for q in used:
        split_counts.setdefault(q.split, np.zeros(n_cls, np.int64))
        split_counts[q.split] += class_counts[q.region]
    weights = class_weights(split_counts["train"])

    meta = {
        "recipe_version": cfg.recipe.version,
        "label_year": cfg.recipe.training_year,
        "class_names": names,
        "class_colors": cfg.classes.colors,
        "ignore_index": cfg.classes.ignore_index,
        "polarisations": cfg.recipe.polarisations,
        "class_weights": weights,
        "regions": [{"name": q.region, "split": q.split, "file": f"{q.region}.npz",
                     "crs": q.crs, "width": q.width, "height": q.height,
                     "class_counts": class_counts[q.region].tolist()} for q in used],
        "qc": [asdict(q) for q in qcs],
        "file_warnings": warnings,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _reports(report_dir, used, class_counts, split_counts, names, stats, qcs)
    log.info("dataset ready: %d regions used, %d excluded -> %s", len(used),
             len(qcs) - len(used), data_dir)
    return meta


def _reports(report_dir: Path, used, class_counts, split_counts, names, stats, qcs) -> None:
    # class distribution per split (pixels and %)
    rows = []
    for split, c in split_counts.items():
        tot = max(int(c.sum()), 1)
        rows += [[split, n, int(v), round(100 * v / tot, 3)] for n, v in zip(names, c,
                                                                           strict=True)]
    _write_csv(report_dir / "class_distribution_by_split.csv",
               ["split", "class", "pixels", "percent"], rows)
    # class distribution per region (%)
    rows = []
    for q in used:
        c = class_counts[q.region]
        tot = max(int(c.sum()), 1)
        rows.append([q.region, q.split] + [round(100 * v / tot, 2) for v in c])
    _write_csv(report_dir / "class_distribution_by_region.csv", ["region", "split", *names], rows)
    # acquisitions per month
    _write_csv(report_dir / "passes_per_month.csv",
               ["region", "split"] + [f"m{m:02d}" for m in range(1, 13)],
               [[q.region, q.split, *q.passes_median] for q in used])
    _write_csv(report_dir / "qc_summary.csv", ["region", "split", "ok", "errors", "warnings"],
               [[q.region, q.split, q.ok, "; ".join(q.errors), "; ".join(q.warnings)]
                for q in qcs])
    (report_dir / "norm_stats.json").write_text(json.dumps(asdict(stats), indent=2))
    try:
        from ..plots import plot_class_distribution
        plot_class_distribution(split_counts, names, report_dir / "class_distribution.png")
    except ImportError:  # matplotlib not installed
        log.info("matplotlib not available; skipping figures")
