"""Evaluation of a trained bundle on validation, test and out-of-distribution regions.

Produces everything the report's results section needs:
* OA, mIoU, per-class IoU / F1 per split (raw and temporally smoothed predictions)
* mIoU per calendar month per split            -> seasonal effects
* confusion matrices                             -> SAR-specific class confusion
* reliability / expected calibration error       -> are the uncertainty maps trustworthy?
* noise floor of class-share changes on stable regions -> minimum detectable change
* class-share error vs. WorldCover, raw and error-adjusted (val confusion applied to test)
* flicker rate (label switches per pixel per year) -> how errors inflate change

The bundle is updated with the validation confusion matrix and the noise floor, which the
prototype uses for error-adjusted percentages and significance flags.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from .analysis.area import error_adjusted_area
from .analysis.metrics import confusion_matrix, for_area_estimation, mean_iou, summary
from .analysis.significance import fit_noise_floor
from .analysis.stats import change_frequency, class_shares, common_valid_mask
from .analysis.temporal import entropy, smooth_probabilities
from .data.dataset import load_regions
from .inference.bundle import load_bundle, write_meta
from .inference.predictor import predict_series

log = logging.getLogger(__name__)
N_BINS = 10


def _labels(probs: np.ndarray, valid: np.ndarray, ignore: int) -> np.ndarray:
    lab = probs.argmax(1).astype(np.uint8)
    lab[~valid] = ignore
    return lab


def evaluate_bundle(bundle_dir: str | Path, data_dir: str | Path, out_dir: str | Path,
                    device: str | None = None, splits=("val", "test", "test_ood"),
                    smoothing_window: int = 3, tile: int = 256, overlap: int = 64) -> dict:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_bundle(bundle_dir)
    model = model.to(device)
    stats, names, ignore, n = meta.norm_stats, meta.class_names, meta.ignore_index, meta.n_classes
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kinds = ("raw", "smoothed")
    cm = {k: {s: np.zeros((n, n), np.int64) for s in splits} for k in kinds}
    cm_month = {s: np.zeros((12, n, n), np.int64) for s in splits}
    cal = {"count": np.zeros(N_BINS), "conf": np.zeros(N_BINS), "acc": np.zeros(N_BINS)}
    ent_ok, ent_bad = [], []
    series = {k: {} for k in kinds}
    ref_shares, flicker = {}, {k: {} for k in kinds}
    region_split = {}

    for split in splits:
        try:
            regions = load_regions(data_dir, (split,))
        except ValueError:
            log.warning("no regions for split %s", split)
            continue
        for r in regions:
            log.info("evaluating %s (%s)", r.name, split)
            region_split[r.name] = split
            probs, valid = predict_series(model, r.db, stats, n, device=device, tile=tile,
                                          overlap=overlap,
                                          temporal_context=meta.temporal_context)
            probs = probs.astype(np.float32)
            labs = {"raw": _labels(probs, valid, ignore),
                    "smoothed": _labels(smooth_probabilities(probs, smoothing_window), valid,
                                        ignore)}
            y = r.label.astype(np.int64)
            for k in kinds:
                for t in range(12):
                    keep = valid[t] & (y != ignore)
                    c = confusion_matrix(y[keep], labs[k][t][keep], n)
                    cm[k][split] += c
                    if k == "raw":
                        cm_month[split][t] += c
            if split == "test":            # calibration + uncertainty on in-distribution test
                for t in range(12):
                    keep = valid[t] & (y != ignore)
                    conf = probs[t].max(0)[keep]
                    ok = labs["raw"][t][keep] == y[keep]
                    b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
                    cal["count"] += np.bincount(b, minlength=N_BINS)
                    cal["conf"] += np.bincount(b, conf, minlength=N_BINS)
                    cal["acc"] += np.bincount(b, ok, minlength=N_BINS)
                    h = entropy(probs[t])[keep]
                    ent_ok.append(float(h[ok].mean()) if ok.any() else np.nan)
                    ent_bad.append(float(h[~ok].mean()) if (~ok).any() else np.nan)
            mask = common_valid_mask(labs["raw"], ignore, valid) & (y != ignore)
            if mask.sum() == 0:
                continue
            ref = np.bincount(y[mask], minlength=n)[:n] / mask.sum() * 100
            ref_shares[r.name] = ref
            for k in kinds:
                series[k][r.name] = class_shares(labs[k], mask, n)
                flicker[k][r.name] = float(change_frequency(labs[k], mask)[mask].mean())

    results: dict = {"bundle": str(bundle_dir), "variant": meta.variant, "splits": {}}
    for s in splits:
        if cm["raw"][s].sum() == 0:
            continue
        results["splits"][s] = {k: summary(cm[k][s], names) for k in kinds}
        results["splits"][s]["miou_per_month"] = [mean_iou(c) for c in cm_month[s]]

    nc = np.maximum(cal["count"], 1)
    ece = float(np.sum(cal["count"] / max(cal["count"].sum(), 1)
                       * np.abs(cal["acc"] / nc - cal["conf"] / nc)))
    results["calibration_test"] = {
        "ece": ece, "bin_confidence": (cal["conf"] / nc).tolist(),
        "bin_accuracy": (cal["acc"] / nc).tolist(), "bin_count": cal["count"].tolist(),
        "mean_entropy_correct": float(np.nanmean(ent_ok)) if ent_ok else None,
        "mean_entropy_wrong": float(np.nanmean(ent_bad)) if ent_bad else None}

    stable = [r for r, s in region_split.items() if s in ("val", "test") and r in series["raw"]]
    floors = {}
    for k in kinds:
        if stable:
            floors[k] = fit_noise_floor([series[k][r] for r in stable], names).__dict__
    results["noise_floor"] = floors
    results["noise_floor_regions"] = stable

    # class-share error vs WorldCover; adjustment uses the VALIDATION confusion (12 months)
    share_rows = []
    cm_val = for_area_estimation(cm["raw"]["val"]) if "val" in cm["raw"] else None
    for r, s in region_split.items():
        if r not in series["raw"]:
            continue
        raw = series["raw"][r]
        err_raw = np.abs(raw - ref_shares[r]).mean(0)
        row = {"region": r, "split": s, "mae_raw_pp": float(err_raw.mean()),
               "mae_smoothed_pp": float(np.abs(series["smoothed"][r] - ref_shares[r])
                                        .mean(0).mean()),
               "flicker_raw": flicker["raw"][r], "flicker_smoothed": flicker["smoothed"][r]}
        if cm_val is not None and cm_val.sum() > 0:
            adj = np.array([100 * error_adjusted_area(cm_val, m / 100).proportion for m in raw])
            row["mae_adjusted_pp"] = float(np.abs(adj - ref_shares[r]).mean(0).mean())
        share_rows.append(row)
    results["share_error"] = share_rows

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    _write_tables(out, results, names)
    _figures(out, results, cm, names)

    # update bundle for the prototype
    if "val" in cm["raw"] and cm["raw"]["val"].sum() > 0:
        meta.confusion_val = cm["raw"]["val"].tolist()
    meta.noise_floor = floors
    meta.test_summary = {s: {"oa": results["splits"][s]["raw"]["overall_accuracy"],
                             "miou": results["splits"][s]["raw"]["mean_iou"]}
                         for s in results["splits"]}
    write_meta(bundle_dir, meta)
    log.info("evaluation written to %s", out)
    return results


def _write_tables(out: Path, results: dict, names: list[str]) -> None:
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "prediction", "class", "iou", "f1", "support_pixels"])
        for s, d in results["splits"].items():
            for k in ("raw", "smoothed"):
                for c in names:
                    pc = d[k]["per_class"][c]
                    w.writerow([s, k, c, round(pc["iou"], 4), round(pc["f1"], 4),
                                pc["support"]])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "prediction", "overall_accuracy", "mean_iou"])
        for s, d in results["splits"].items():
            for k in ("raw", "smoothed"):
                w.writerow([s, k, round(d[k]["overall_accuracy"], 4),
                            round(d[k]["mean_iou"], 4)])
    if results["share_error"]:
        keys = list(results["share_error"][0])
        with open(out / "share_error.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results["share_error"])


def _figures(out: Path, results: dict, cm: dict, names: list[str]) -> None:
    try:
        from .plots import plot_confusion, plot_monthly_metric, plot_reliability
    except ImportError:
        return
    for s in results["splits"]:
        plot_confusion(cm["raw"][s], names, out / f"confusion_{s}.png",
                       f"{results['variant']} - {s} (all months)")
    plot_monthly_metric({s: d["miou_per_month"] for s, d in results["splits"].items()},
                        out / "miou_per_month.png", "mIoU", "mIoU by calendar month")
    c = results["calibration_test"]
    if sum(c["bin_count"]):
        plot_reliability(np.array(c["bin_confidence"]), np.array(c["bin_accuracy"]),
                         np.array(c["bin_count"]), c["ece"], out / "reliability_test.png")


def compare_variants(results_root: str | Path, train_root: str | Path) -> list[dict]:
    """Collect one row per variant (best epoch, val/test/OOD scores) into variants.csv."""
    rows = []
    for mpath in sorted(Path(results_root).glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        v = m["variant"]
        hist_path = Path(train_root) / v / "history.json"
        hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
        row = {"variant": v, "epochs_run": len(hist),
               "best_epoch": max(hist, key=lambda h: h["val_miou"])["epoch"] if hist else None,
               "train_minutes": round(sum(h["seconds"] for h in hist) / 60, 1)}
        for s, d in m["splits"].items():
            row[f"{s}_oa"] = round(d["raw"]["overall_accuracy"], 4)
            row[f"{s}_miou"] = round(d["raw"]["mean_iou"], 4)
            row[f"{s}_miou_smoothed"] = round(d["smoothed"]["mean_iou"], 4)
        rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        with open(Path(results_root) / "variants.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows
