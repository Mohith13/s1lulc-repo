"""Is a measured change real, or within the model's own noise?

Method: run the full pipeline on areas/periods where land cover is (assumed) stable, e.g.
the test regions in the label year. Every change measured there is error, not land-use
change. The q-th percentile of |change| per class is the *noise floor* (minimum detectable
change). A change in a new area is flagged significant only if it exceeds that floor.

Month-to-month and first-to-last floors are kept separately: seasonal confusion produces
large monthly swings but largely cancels between month 1 and month 12.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .stats import month_to_month, total_change


@dataclass
class NoiseFloor:
    class_names: list[str]
    month_step_pp: list[float]   # per class, q-th percentile of |monthly change| on stable areas
    total_pp: list[float]        # per class, q-th percentile of |first-to-last change|
    quantile: float
    n_series: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NoiseFloor:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_noise_floor(stable_share_series: list[np.ndarray], class_names: list[str],
                    quantile: float = 95.0) -> NoiseFloor:
    """stable_share_series: list of (T, C) share arrays (%) from no-change areas."""
    if not stable_share_series:
        raise ValueError("need at least one stable series")
    steps = np.concatenate([np.abs(month_to_month(s)) for s in stable_share_series], axis=0)
    totals = np.stack([np.abs(total_change(s)) for s in stable_share_series], axis=0)
    return NoiseFloor(
        class_names=list(class_names),
        month_step_pp=np.percentile(steps, quantile, axis=0).tolist(),
        total_pp=np.percentile(totals, quantile, axis=0).tolist(),
        quantile=float(quantile),
        n_series=len(stable_share_series),
    )


def flag_significant(changes_pp: np.ndarray, floor_pp: np.ndarray | list[float]) -> np.ndarray:
    """Boolean array (same shape as changes): True where |change| exceeds the class floor."""
    return np.abs(np.asarray(changes_pp)) > np.asarray(floor_pp)
