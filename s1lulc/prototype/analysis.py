"""Prototype analysis: everything computed from the model's monthly probabilities.

Pure function of arrays (no Earth Engine, no files), so it is fully unit-tested.

For both the raw monthly maps and the temporally smoothed maps:
* class shares per month on the common valid area (percent)
* month-to-month change and total change, first to last month (percentage points)
* significance of every change against the model's noise floor (from the bundle)
* error-adjusted shares with 95% confidence intervals (Olofsson et al. 2014)
* transition matrix and gross vs net change, first to last month
Plus per-pixel uncertainty (entropy), label flicker, and agreement with Dynamic World.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..analysis.area import adjusted_share_series
from ..analysis.metrics import confusion_matrix, for_area_estimation
from ..analysis.significance import flag_significant
from ..analysis.stats import change_frequency, class_shares, gross_net, transition_matrix
from ..analysis.temporal import entropy, smooth_probabilities

MISSING_MONTH_VALID_FRACTION = 0.01
LOW_PASSES = 3
DEFAULT_EFFECTIVE_N = 500          # assumed independent reference samples per class (see area.py)


def _labels(probs: np.ndarray, valid: np.ndarray, ignore: int) -> np.ndarray:
    lab = probs.argmax(1).astype(np.uint8)
    lab[~valid] = ignore
    return lab


def _series(labels: np.ndarray, mask: np.ndarray, available: list[bool], n: int) -> np.ndarray:
    shares = np.full((len(available), n), np.nan)
    sub = class_shares(labels[available], mask, n)
    shares[np.asarray(available)] = sub
    return shares


def _changes(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Month-to-month (NaN if either month missing) and total (last minus first available)."""
    m2m = np.diff(shares, axis=0)
    avail = np.flatnonzero(~np.isnan(shares[:, 0]))
    first, last = int(avail[0]), int(avail[-1])
    return m2m, shares[last] - shares[first], first, last


def analyse(probs: np.ndarray, valid: np.ndarray, class_names: list[str], ignore: int,
            confusion_val: list | np.ndarray | None = None, noise_floor: dict | None = None,
            passes: np.ndarray | None = None, dw_labels: np.ndarray | None = None,
            smoothing_window: int = 3, effective_n: float = DEFAULT_EFFECTIVE_N,
            platforms: list[list[str]] | None = None, scenes: list[int] | None = None) -> dict:
    """probs (12, C, H, W), valid (12, H, W) -> results dict (arrays + JSON-ready tables)."""
    probs = np.asarray(probs, np.float32)
    t, n = probs.shape[:2]
    valid = np.asarray(valid, bool)
    vfrac = valid.reshape(t, -1).mean(1)
    available = [bool(f >= MISSING_MONTH_VALID_FRACTION) for f in vfrac]
    if sum(available) < 2:
        raise ValueError("fewer than two months contain Sentinel-1 data for this location")
    mask = np.all(valid[available], axis=0)
    if not mask.any():
        raise ValueError("no pixel has data in every available month")

    smoothed = smooth_probabilities(probs, smoothing_window) if smoothing_window > 1 else probs
    out: dict = {"class_names": class_names, "n_months": t, "available": available,
                 "valid_fraction": vfrac.round(4).tolist(),
                 "common_valid_fraction": float(mask.mean()), "smoothing_window": smoothing_window,
                 "maps": {}, "series": {}, "warnings": []}

    cm_ref = None
    if confusion_val is not None and np.asarray(confusion_val).sum() > 0:
        cm_ref = for_area_estimation(np.asarray(confusion_val))     # map x reference
    for kind, p in (("raw", probs), ("smoothed", smoothed)):
        lab = _labels(p, valid, ignore)
        out["maps"][kind] = lab
        shares = _series(lab, mask, available, n)
        m2m, total, first, last = _changes(shares)
        s = {"shares": shares, "month_to_month": m2m, "total": total,
             "first_month": first, "last_month": last}
        floor = (noise_floor or {}).get(kind)
        if floor:
            s["significant_m2m"] = flag_significant(np.nan_to_num(m2m), floor["month_step_pp"])
            s["significant_total"] = flag_significant(total, floor["total_pp"])
            s["floor_m2m"] = np.asarray(floor["month_step_pp"])
            s["floor_total"] = np.asarray(floor["total_pp"])
        if cm_ref is not None:
            adj = np.full_like(shares, np.nan)
            ci = np.full_like(shares, np.nan)
            a, c = adjusted_share_series(cm_ref, shares[available], effective_n)
            adj[np.asarray(available)], ci[np.asarray(available)] = a, c
            s["adjusted"], s["adjusted_ci95"] = adj, ci
        trans = transition_matrix(lab[first], lab[last], mask, n)
        g = gross_net(trans)
        s.update(transitions=trans, gross_gain=g.gain, gross_loss=g.loss,
                 changed_total=g.changed_total)
        flick = change_frequency(lab[available], mask)
        s["flicker_mean"] = float(flick[mask].mean())
        out["maps"][f"{kind}_change_frequency"] = flick
        out["series"][kind] = s

    ent = np.stack([entropy(probs[i]) if available[i] else np.full(mask.shape, np.nan, np.float32)
                    for i in range(t)])
    ent[:, ~mask] = np.nan
    with warnings.catch_warnings():               # pixels outside the mask are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        out["maps"]["entropy_mean"] = np.nanmean(ent[available], axis=0).astype(np.float32)
    out["entropy_per_month"] = [float(np.nanmean(e)) if a else None
                                for e, a in zip(ent, available, strict=True)]
    conf = probs.max(1)
    out["low_confidence_fraction"] = [float((conf[i][mask] < 0.5).mean()) if available[i]
                                      else None for i in range(t)]

    if passes is not None:
        pm = [int(np.median(p[p > 0])) if np.any(p > 0) else 0 for p in passes]
        out["passes_median"] = pm
        low = [i for i, p in enumerate(pm) if available[i] and p < LOW_PASSES]
        if low:
            out["warnings"].append({"kind": "few_acquisitions", "months": low,
                                    "text": "fewer than 3 Sentinel-1 acquisitions: noisier "
                                            "composite, less reliable map"})
    if platforms is not None:
        out["platforms"] = [sorted(p) for p in platforms]
        out["scenes"] = list(scenes) if scenes is not None else None
        sw = sensor_changes(platforms, available)
        if sw:
            out["warnings"].append({
                "kind": "sensor_change", "months": sw,
                "text": "the set of Sentinel-1 satellites changes in these months; radiometric "
                        "and orbit-timing differences can appear as land-cover change"})
    missing = [i for i, a in enumerate(available) if not a]
    if missing:
        out["warnings"].append({"kind": "missing_month", "months": missing,
                                "text": "no Sentinel-1 data; month excluded from statistics"})
    if float(np.nanmean(out["maps"]["entropy_mean"][mask])) > 0.5:
        out["warnings"].append({"kind": "high_uncertainty", "months": [],
                                "text": "high average uncertainty: this area may be unlike the "
                                        "training regions; treat class shares with caution"})

    if dw_labels is not None:
        out["dynamic_world"] = _dw_compare(out["maps"]["raw"], dw_labels, mask, available, n,
                                           ignore)
    return out


def sensor_changes(platforms: list[list[str]], available: list[bool]) -> list[int]:
    """Indices of months whose satellite set differs from the previous available month."""
    out, prev = [], None
    for i, (p, a) in enumerate(zip(platforms, available, strict=True)):
        if not a or not p:
            continue
        cur = frozenset(p)
        if prev is not None and cur != prev:
            out.append(i)
        prev = cur
    return out


def _dw_compare(ours: np.ndarray, dw: np.ndarray, mask: np.ndarray, available: list[bool],
                n: int, ignore: int) -> dict:
    """Monthly agreement with Dynamic World (already mapped to our classes; 255 = no data)."""
    agree, coverage, shares = [], [], np.full((len(available), n), np.nan)
    cm = np.zeros((n, n), np.int64)
    for i, a in enumerate(available):
        both = mask & (dw[i] != ignore) & (ours[i] != ignore)
        coverage.append(float(both.sum() / max(mask.sum(), 1)))
        if not a or both.sum() < 0.05 * mask.sum():
            agree.append(None)
            continue
        agree.append(float((ours[i][both] == dw[i][both]).mean()))
        shares[i] = np.bincount(dw[i][both], minlength=n)[:n] / both.sum() * 100
        cm += confusion_matrix(dw[i][both], ours[i][both], n)
    vals = [x for x in agree if x is not None]
    return {"agreement": agree, "coverage": coverage, "shares": shares,
            "confusion_dw_vs_ours": cm, "mean_agreement": float(np.mean(vals)) if vals else None}
