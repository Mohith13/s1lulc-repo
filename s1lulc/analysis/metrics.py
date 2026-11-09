"""Segmentation accuracy metrics from a confusion matrix.

Convention used throughout the project: confusion[i, j] = number of pixels with
REFERENCE (label) class i and PREDICTED class j. (Note: area.py takes the transpose,
rows = map/predicted, as in Olofsson et al.; use ``for_area_estimation``.)
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                     ignore_index: int | None = None) -> np.ndarray:
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    keep = (t >= 0) & (t < n_classes) & (p >= 0) & (p < n_classes)
    if ignore_index is not None:
        keep &= t != ignore_index
    return np.bincount(t[keep] * n_classes + p[keep],
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def overall_accuracy(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.float64)
    return float(np.trace(cm) / max(cm.sum(), 1))


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU_c = TP / (TP + FP + FN); NaN for a class absent from both labels and predictions."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def mean_iou(cm: np.ndarray) -> float:
    """Mean over classes present in labels or predictions (NaNs skipped)."""
    return float(np.nanmean(per_class_iou(cm)))


def for_area_estimation(cm: np.ndarray) -> np.ndarray:
    """Reference x predicted -> predicted (map) x reference, as area.py expects."""
    return np.asarray(cm).T


def summary(cm: np.ndarray, class_names: list[str]) -> dict:
    iou, f1 = per_class_iou(cm), per_class_f1(cm)
    return {
        "overall_accuracy": overall_accuracy(cm),
        "mean_iou": mean_iou(cm),
        "per_class": {n: {"iou": float(i), "f1": float(f), "support": int(s)}
                      for n, i, f, s in zip(class_names, iou, f1, np.asarray(cm).sum(1),
                                            strict=True)},
    }
