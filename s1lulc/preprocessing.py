"""Client-side preprocessing shared by training and the prototype.

Earth Engine returns monthly composites in LINEAR power (averaging in linear units is the
physically correct mean; averaging dB values would give a geometric mean biased low). Here we:

1. convert linear power to dB,
2. clip to percentiles computed on the training regions,
3. z-score with training mean/std,
4. replace invalid pixels with 0 (the normalised mean) and return a validity mask.

The statistics are saved with the model bundle so inference uses exactly the training values.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EPS_LINEAR = 1e-6  # floor before log10 (-60 dB), avoids log(0)


def linear_to_db(x: np.ndarray) -> np.ndarray:
    """10*log10(x) with NaN preserved and non-positive values floored."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    finite = np.isfinite(x)
    out[finite] = 10.0 * np.log10(np.maximum(x[finite], EPS_LINEAR))
    return out.astype(np.float32)


def valid_mask(stack: np.ndarray) -> np.ndarray:
    """True where every band of a (bands, H, W) stack is finite and positive."""
    stack = np.asarray(stack)
    return np.all(np.isfinite(stack) & (stack > 0), axis=0)


@dataclass
class NormStats:
    """Per-band normalisation statistics in dB, computed on training regions only."""

    bands: list[str]
    clip_low: list[float]
    clip_high: list[float]
    mean: list[float]
    std: list[float]
    recipe_version: str

    @classmethod
    def fit(cls, db_samples: np.ndarray, bands: list[str], percentiles: tuple[float, float],
            recipe_version: str) -> NormStats:
        """Fit from an array of shape (n_bands, n_pixels) in dB; NaNs are ignored."""
        db_samples = np.asarray(db_samples, dtype=np.float64)
        if db_samples.ndim != 2 or db_samples.shape[0] != len(bands):
            raise ValueError("db_samples must have shape (n_bands, n_pixels)")
        lo, hi, mu, sd = [], [], [], []
        for b in db_samples:
            b = b[np.isfinite(b)]
            if b.size == 0:
                raise ValueError("no finite samples for a band")
            p_lo, p_hi = np.percentile(b, percentiles)
            c = np.clip(b, p_lo, p_hi)
            lo.append(float(p_lo))
            hi.append(float(p_hi))
            mu.append(float(c.mean()))
            sd.append(float(max(c.std(), 1e-6)))
        return cls(bands=list(bands), clip_low=lo, clip_high=hi, mean=mu, std=sd,
                   recipe_version=recipe_version)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def normalise_db(db: np.ndarray, stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    """(bands, H, W) backscatter in dB (NaN = invalid) -> (normalised float32, valid (H, W)).

    This is the single function that prepares model input; training and inference both
    call it, so the scaling can never differ between the two.
    """
    db = np.asarray(db, dtype=np.float32)
    if db.shape[0] != len(stats.bands):
        raise ValueError(f"expected {len(stats.bands)} bands, got {db.shape[0]}")
    valid = np.all(np.isfinite(db), axis=0)
    lo = np.asarray(stats.clip_low, np.float32)[:, None, None]
    hi = np.asarray(stats.clip_high, np.float32)[:, None, None]
    mu = np.asarray(stats.mean, np.float32)[:, None, None]
    sd = np.asarray(stats.std, np.float32)[:, None, None]
    out = (np.clip(np.nan_to_num(db, nan=0.0), lo, hi) - mu) / sd
    out[:, ~valid] = 0.0
    return out.astype(np.float32), valid


def month_indices(t: int, n_months: int, temporal_context: int) -> list[int]:
    """Months used as input for month ``t``: t-k .. t+k, clipped to the series (edges repeat)."""
    return [min(max(t + o, 0), n_months - 1)
            for o in range(-temporal_context, temporal_context + 1)]


def model_input(db_series: np.ndarray, t: int, stats: NormStats,
                temporal_context: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build the network input for month ``t`` from a (T, 2, H, W) dB series.

    temporal_context = k feeds months t-k .. t+k (oldest first) as 2*(2k+1) channels. Months
    outside the series are replaced by the nearest available month (edge repetition), and
    neighbour pixels without data fall back to month t, so a gap in a neighbouring month never
    removes a pixel. Validity is defined by month t alone. Training, evaluation and the
    prototype all call this function.
    """
    db_series = np.asarray(db_series)
    centre = db_series[t].astype(np.float32)
    x_c, valid = normalise_db(centre, stats)
    if temporal_context == 0:
        return x_c, valid
    parts = []
    for j in month_indices(t, db_series.shape[0], temporal_context):
        if j == t:
            parts.append(x_c)
            continue
        nb = db_series[j].astype(np.float32)
        nb = np.where(np.all(np.isfinite(nb), axis=0, keepdims=True), nb, centre)
        parts.append(normalise_db(nb, stats)[0])
    x = np.concatenate(parts)
    x[:, ~valid] = 0.0
    return x, valid


def linear_stack_to_db(linear: np.ndarray) -> np.ndarray:
    """Linear power -> dB with every invalid (non-finite or <= 0) pixel set to NaN."""
    linear = np.asarray(linear)
    db = linear_to_db(linear)
    db[~(np.isfinite(linear) & (linear > 0))] = np.nan
    return db


def normalise(linear: np.ndarray, stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    """(bands, H, W) linear power -> (normalised float32 array, valid mask (H, W))."""
    return normalise_db(linear_stack_to_db(linear), stats)
