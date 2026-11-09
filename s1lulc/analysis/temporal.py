"""Uncertainty and temporal consistency of monthly predictions.

* ``entropy`` / ``confidence``: per-pixel uncertainty from softmax probabilities.
* ``smooth_probabilities``: moving average of class probabilities over neighbouring months
  before taking the argmax. Real land-use change is rare and persistent; one-month flips are
  mostly seasonal confusion or noise, so smoothing suppresses them. The prototype reports raw
  and smoothed results side by side so the effect is visible, not hidden.
"""

from __future__ import annotations

import numpy as np


def confidence(probs: np.ndarray, axis: int = -3) -> np.ndarray:
    """Top-1 probability. probs: (..., C, H, W) by default."""
    return np.max(probs, axis=axis)


def entropy(probs: np.ndarray, axis: int = -3, normalised: bool = True) -> np.ndarray:
    """Shannon entropy over the class axis; normalised to 0..1 by log(C) if requested."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=axis)
    if normalised:
        h = h / np.log(p.shape[axis])
    return h.astype(np.float32)


def smooth_probabilities(probs: np.ndarray, window: int = 3) -> np.ndarray:
    """(T, C, H, W) probabilities -> same shape, centred moving average over time.

    Edges use the available months only (no wrap-around: December and the following January
    are a year apart in the prototype's window, not neighbours).
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 4:
        raise ValueError("probs must have shape (T, C, H, W)")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1:
        return probs.copy()
    t, half = probs.shape[0], window // 2
    csum = np.concatenate([np.zeros_like(probs[:1]), np.cumsum(probs, axis=0)], axis=0)
    out = np.empty_like(probs)
    for i in range(t):
        lo, hi = max(0, i - half), min(t, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def labels_from_probs(probs: np.ndarray, valid: np.ndarray, ignore_index: int) -> np.ndarray:
    """(T, C, H, W) -> (T, H, W) uint8 labels; invalid pixels set to ignore_index."""
    labels = np.argmax(probs, axis=1).astype(np.uint8)
    labels[:, ~np.asarray(valid, bool)] = ignore_index
    return labels
