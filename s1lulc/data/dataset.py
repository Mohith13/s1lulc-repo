"""PyTorch datasets over the built region files.

Training samples are drawn on the fly: a random region, a random 256x256 crop and a random
month. Using a random month per sample is free seasonal augmentation (the WorldCover label
is annual, so the model must recognise each class in every season).

Rare classes: with probability ``rare_crop_prob`` the crop is centred on a pixel of a rare
class (train share < ``rare_threshold``), so wetland or bare land is seen often enough to
be learned. This and the class-weighted loss are the two class-imbalance measures.

Augmentation (in dB, before normalisation): flips and 90-degree rotations, a random gain
offset (calibration / incidence-angle variation) and additive dB noise (residual speckle;
months with few acquisitions are noisier). Flips/rotations are not physically exact for
SAR look geometry; this is a documented approximation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..preprocessing import NormStats, model_input, month_indices


@dataclass
class RegionData:
    name: str
    split: str
    db: np.ndarray        # (12, 2, H, W) float16, NaN = no data
    label: np.ndarray     # (H, W) uint8
    count: np.ndarray     # (12, H, W) uint8


@dataclass
class AugmentConfig:
    flip: bool = True
    rot90: bool = True
    gain_db: float = 1.0
    noise_db: float = 0.5
    noise_prob: float = 0.5


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def load_regions(data_dir: str | Path, splits: tuple[str, ...]) -> list[RegionData]:
    data_dir = Path(data_dir)
    out = []
    for r in load_meta(data_dir)["regions"]:
        if r["split"] in splits:
            with np.load(data_dir / r["file"]) as z:
                out.append(RegionData(r["name"], r["split"], z["db"], z["label"], z["count"]))
    if not out:
        raise ValueError(f"no regions for splits {splits} in {data_dir}")
    return out


class RandomPatchDataset:
    """Map-style dataset of ``samples_per_epoch`` random patches (torch-compatible)."""

    def __init__(self, regions: list[RegionData], stats: NormStats, n_classes: int,
                 ignore_index: int, patch_size: int = 256, samples_per_epoch: int = 2000,
                 rare_crop_prob: float = 0.3, rare_threshold: float = 0.05,
                 augment: AugmentConfig | None = None, seed: int = 42,
                 max_rare_index: int = 20_000, temporal_context: int = 0):
        self.regions, self.stats = regions, stats
        self.n_classes, self.ignore = n_classes, ignore_index
        self.p, self.n = patch_size, samples_per_epoch
        self.rare_crop_prob = rare_crop_prob
        self.aug = augment or AugmentConfig()
        self.seed, self.epoch = seed, 0
        self.k = temporal_context
        for r in regions:
            if min(r.label.shape) < patch_size:
                raise ValueError(f"{r.name}: region smaller than patch size {patch_size}")

        counts = np.zeros(n_classes, np.int64)
        for r in regions:
            lab = r.label[r.label < n_classes]
            counts += np.bincount(lab, minlength=n_classes)[:n_classes]
        share = counts / max(counts.sum(), 1)
        self.rare_classes = [c for c in range(n_classes) if 0 < share[c] < rare_threshold]

        # rare-class pixel index: {class: [(region_idx, flat_indices), ...]}
        rng = np.random.default_rng(seed)
        self.rare_index: dict[int, list[tuple[int, np.ndarray]]] = {}
        for c in self.rare_classes:
            for i, r in enumerate(regions):
                idx = np.flatnonzero(r.label == c)
                if idx.size:
                    if idx.size > max_rare_index:
                        idx = rng.choice(idx, max_rare_index, replace=False)
                    self.rare_index.setdefault(c, []).append((i, idx))

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so every epoch draws different (but reproducible) patches."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def _crop_origin(self, rng, region: RegionData, centre=None) -> tuple[int, int]:
        h, w = region.label.shape
        if centre is None:
            return int(rng.integers(0, h - self.p + 1)), int(rng.integers(0, w - self.p + 1))
        cy, cx = centre
        y = int(np.clip(cy - self.p // 2 + rng.integers(-self.p // 4, self.p // 4 + 1),
                        0, h - self.p))
        x = int(np.clip(cx - self.p // 2 + rng.integers(-self.p // 4, self.p // 4 + 1),
                        0, w - self.p))
        return y, x

    def sample(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x float32 (2*(2k+1), P, P) normalised, y int64 (P, P))."""
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if self.rare_index and rng.random() < self.rare_crop_prob:
            c = self.rare_classes[rng.integers(len(self.rare_classes))]
            choices = self.rare_index.get(c)
            if choices:
                ri, idx = choices[rng.integers(len(choices))]
                region = self.regions[ri]
                flat = int(idx[rng.integers(idx.size)])
                y0, x0 = self._crop_origin(rng, region, divmod(flat, region.label.shape[1]))
            else:
                region = self.regions[rng.integers(len(self.regions))]
                y0, x0 = self._crop_origin(rng, region)
        else:
            region = self.regions[rng.integers(len(self.regions))]
            y0, x0 = self._crop_origin(rng, region)

        sl = (slice(y0, y0 + self.p), slice(x0, x0 + self.p))
        n_months = region.db.shape[0]
        for _ in range(4):                      # avoid months with no data in this crop
            month = int(rng.integers(n_months))
            if np.isfinite(region.db[month][:, sl[0], sl[1]]).all(axis=0).mean() > 0.5:
                break
        idx = month_indices(month, n_months, self.k)          # e.g. [m-1, m, m+1]
        db = region.db[idx][:, :, sl[0], sl[1]].astype(np.float32)   # (2k+1, 2, P, P)
        y = region.label[sl].astype(np.int64)

        if self.aug.gain_db:                    # one offset: calibration / angle variation
            db = db + rng.uniform(-self.aug.gain_db, self.aug.gain_db)
        if self.aug.noise_db and rng.random() < self.aug.noise_prob:
            db = db + rng.normal(0, self.aug.noise_db, size=db.shape).astype(np.float32)
        x, valid = model_input(db, self.k, self.stats, self.k)
        y[~valid] = self.ignore

        if self.aug.rot90:
            k = int(rng.integers(4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if self.aug.flip and rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if self.aug.flip and rng.random() < 0.5:
            x, y = x[:, ::-1, :], y[::-1, :]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def __getitem__(self, i: int):
        import torch

        x, y = self.sample(i)
        return torch.from_numpy(x), torch.from_numpy(y)
