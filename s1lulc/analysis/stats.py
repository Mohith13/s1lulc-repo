"""Class shares and change statistics for a sequence of monthly label maps.

Definitions (all in percent of the COMMON valid area, i.e. pixels valid in every month):

* share      P_m(c)   = 100 * n_m(c) / N_valid
* month step dP_m(c)  = P_m(c) - P_{m-1}(c)            [percentage points, pp]
* total      dP(c)    = P_last(c) - P_first(c)         [pp]
* transition T[i, j]  = 100 * #(first == i and last == j) / N_valid
* gross gain/loss     = pixels entering / leaving class c between first and last month

Using one common mask matters: if month 3 had a nodata stripe, a per-month denominator
would turn missing data into apparent land-cover change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def common_valid_mask(maps: np.ndarray, ignore_index: int,
                      extra_valid: np.ndarray | None = None) -> np.ndarray:
    """(T, H, W) label maps -> (H, W) mask of pixels valid in every month."""
    maps = np.asarray(maps)
    if maps.ndim != 3:
        raise ValueError("maps must have shape (T, H, W)")
    valid = np.all(maps != ignore_index, axis=0)
    if extra_valid is not None:
        extra_valid = np.asarray(extra_valid, bool)
        valid &= extra_valid if extra_valid.ndim == 2 else np.all(extra_valid, axis=0)
    return valid


def class_shares(maps: np.ndarray, valid: np.ndarray, n_classes: int) -> np.ndarray:
    """(T, H, W) maps + (H, W) mask -> (T, C) shares in percent. Rows sum to 100."""
    maps = np.asarray(maps)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no pixel is valid in all months; cannot compute shares")
    out = np.empty((maps.shape[0], n_classes), dtype=np.float64)
    for t in range(maps.shape[0]):
        counts = np.bincount(maps[t][valid].astype(np.int64), minlength=n_classes)[:n_classes]
        out[t] = 100.0 * counts / n_valid
    return out


def month_to_month(shares: np.ndarray) -> np.ndarray:
    """(T, C) shares -> (T-1, C) changes in percentage points."""
    return np.diff(np.asarray(shares, dtype=np.float64), axis=0)


def total_change(shares: np.ndarray) -> np.ndarray:
    """(T, C) shares -> (C,) last minus first month, in percentage points."""
    shares = np.asarray(shares, dtype=np.float64)
    return shares[-1] - shares[0]


def transition_matrix(first: np.ndarray, last: np.ndarray, valid: np.ndarray,
                      n_classes: int) -> np.ndarray:
    """T[i, j] = percent of valid area that is class i in ``first`` and class j in ``last``."""
    a = first[valid].astype(np.int64)
    b = last[valid].astype(np.int64)
    m = np.bincount(a * n_classes + b, minlength=n_classes * n_classes)
    return 100.0 * m.reshape(n_classes, n_classes) / max(int(valid.sum()), 1)


@dataclass
class GrossNet:
    gain: np.ndarray        # (C,) pp of area that became class c
    loss: np.ndarray        # (C,) pp of area that stopped being class c
    net: np.ndarray         # (C,) gain - loss == total change
    changed_total: float    # pp of area whose label differs between first and last month


def gross_net(transitions: np.ndarray) -> GrossNet:
    t = np.asarray(transitions, dtype=np.float64)
    diag = np.diag(t)
    gain = t.sum(axis=0) - diag
    loss = t.sum(axis=1) - diag
    return GrossNet(gain=gain, loss=loss, net=gain - loss,
                    changed_total=float(t.sum() - diag.sum()))


def change_frequency(maps: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(H, W) number of month-to-month label switches per pixel (0 = stable all year).

    High counts on a pixel usually mean classification flicker, not real land-use change.
    """
    maps = np.asarray(maps)
    switches = (maps[1:] != maps[:-1]).sum(axis=0).astype(np.int16)
    switches[~valid] = -1
    return switches
