"""Error-adjusted area estimation (Olofsson et al., 2014, Remote Sensing of Environment 148).

Raw pixel counting is biased whenever the classifier confuses classes: if 10% of true cropland
is predicted as grassland, the mapped cropland share is too low. The stratified estimator
corrects mapped proportions with an accuracy-assessment confusion matrix.

Notation: n[i, j] = number of reference samples with MAP class i and REFERENCE class j
(rows = map, columns = reference). W[i] = mapped area proportion of class i in the area of
interest (sum to 1).

    p_hat[i, j] = W[i] * n[i, j] / n[i, .]
    area proportion of class j:  p_.j = sum_i p_hat[i, j]
    SE(p_.j) = sqrt( sum_i W[i]^2 * (n[i,j]/n[i,.]) * (1 - n[i,j]/n[i,.]) / (n[i,.] - 1) )

Caveats documented in the report:
* Our 'reference' is WorldCover (~77% accurate), not field truth.
* The confusion matrix comes from the test regions; applying it to a new location assumes
  errors transfer. That is weakest for out-of-distribution regions.
* Pixel-level counts are spatially autocorrelated, so treating every pixel as an independent
  sample makes the SE far too small. ``effective_n`` rescales each row to a chosen number of
  effectively independent samples before computing the SE (a deliberate, stated assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054


@dataclass
class AreaEstimate:
    proportion: np.ndarray   # (C,) error-adjusted area proportion, 0..1
    se: np.ndarray           # (C,) standard error of the proportion
    mapped: np.ndarray       # (C,) raw mapped proportion W, 0..1

    @property
    def ci95(self) -> np.ndarray:
        """(C,) half-width of the 95% confidence interval, as a proportion."""
        return Z95 * self.se

    def as_percent(self) -> dict[str, np.ndarray]:
        return {"adjusted": 100 * self.proportion, "ci95": 100 * self.ci95,
                "mapped": 100 * self.mapped}


def error_adjusted_area(confusion: np.ndarray, mapped_props: np.ndarray,
                        effective_n: float | None = None) -> AreaEstimate:
    """Olofsson stratified estimator.

    confusion: (C, C) counts, rows = map class, columns = reference class.
    mapped_props: (C,) mapped proportions W in the target area (sum to 1; percent also OK).
    effective_n: optional number of effectively independent samples per map class.
    """
    n = np.asarray(confusion, dtype=np.float64)
    w = np.asarray(mapped_props, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] != n.shape[1] or w.shape != (n.shape[0],):
        raise ValueError("confusion must be (C, C) and mapped_props (C,)")
    if np.any(n < 0) or np.any(w < 0):
        raise ValueError("counts and proportions must be non-negative")
    if w.sum() > 1.0 + 1e-6:          # accept percentages
        w = w / 100.0 if abs(w.sum() - 100) < 1e-3 else w / w.sum()

    row = n.sum(axis=1)
    # Rows without samples cannot be corrected; keep that class's mapped area as-is
    # (identity row), which is the conservative choice, and give it zero variance.
    safe = row > 0
    q = np.zeros_like(n)
    q[safe] = n[safe] / row[safe, None]
    q[~safe] = np.eye(n.shape[0])[~safe]

    p = (w[:, None] * q).sum(axis=0)

    n_eff = row.copy()
    if effective_n is not None:
        if effective_n <= 1:
            raise ValueError("effective_n must be > 1")
        n_eff = np.where(safe, np.minimum(row, effective_n), 0)
    denom = np.where(n_eff > 1, n_eff - 1, np.inf)
    var = ((w[:, None] ** 2) * q * (1 - q) / denom[:, None]).sum(axis=0)
    return AreaEstimate(proportion=p, se=np.sqrt(var), mapped=w)


def adjusted_share_series(confusion: np.ndarray, shares_pct: np.ndarray,
                          effective_n: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply the estimator month by month. Returns (adjusted %, ci95 %), each (T, C)."""
    adj, ci = [], []
    for s in np.asarray(shares_pct, dtype=np.float64):
        est = error_adjusted_area(confusion, s / 100.0, effective_n)
        adj.append(100 * est.proportion)
        ci.append(100 * est.ci95)
    return np.array(adj), np.array(ci)
