"""Typed access to the YAML configuration files.

All settings live in ``configs/``; code never hard-codes class lists, recipe parameters or
region coordinates. This keeps training, evaluation and the prototype consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
VALID_SPLITS = {"train", "val", "test", "test_ood"}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ClassScheme:
    names: list[str]
    colors: list[str]
    ignore_index: int
    worldcover_to_target: dict[int, int]
    dynamicworld_to_target: dict[int, int]

    @property
    def n_classes(self) -> int:
        return len(self.names)

    @staticmethod
    def _lut(mapping: dict[int, int], ignore: int) -> np.ndarray:
        """Lookup table: lut[source_code] = target id; unknown codes -> ignore."""
        lut = np.full(256, ignore, dtype=np.uint8)
        for src, dst in mapping.items():
            lut[int(src)] = int(dst)
        return lut

    def map_worldcover(self, codes: np.ndarray) -> np.ndarray:
        """Raw WorldCover codes (any shape) -> target class ids; NaN/out of range -> ignore."""
        return self._map(codes, self.worldcover_to_target)

    def map_dynamicworld(self, labels: np.ndarray) -> np.ndarray:
        return self._map(labels, self.dynamicworld_to_target)

    def _map(self, values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
        values = np.asarray(values)
        lut = self._lut(mapping, self.ignore_index)
        out = np.full(values.shape, self.ignore_index, dtype=np.uint8)
        finite = np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, bool)
        ints = np.where(finite, values, 0).astype(np.int64)
        in_range = finite & (ints >= 0) & (ints < 256)
        out[in_range] = lut[ints[in_range]]
        return out


@dataclass(frozen=True)
class Recipe:
    version: str
    collection: str
    instrument_mode: str
    polarisations: list[str]
    orbit_pass: str
    edge_mask_linear: float
    reducer: str
    pixel_size_m: int
    aoi_half_size_m: int
    training_year: int
    label_collection: str
    db_clip_percentiles: tuple[float, float]

    def __post_init__(self) -> None:
        if self.orbit_pass not in {"both", "ASCENDING", "DESCENDING"}:
            raise ValueError(f"orbit_pass must be both/ASCENDING/DESCENDING, got {self.orbit_pass}")
        if self.reducer not in {"mean", "median"}:
            raise ValueError(f"reducer must be mean or median, got {self.reducer}")
        if self.aoi_half_size_m % self.pixel_size_m:
            raise ValueError("aoi_half_size_m must be a multiple of pixel_size_m")


@dataclass(frozen=True)
class Region:
    name: str
    lon: float
    lat: float
    split: str
    note: str = ""

    def __post_init__(self) -> None:
        if not -180 <= self.lon <= 180 or not -90 <= self.lat <= 90:
            raise ValueError(f"{self.name}: invalid coordinates lon={self.lon} lat={self.lat}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"{self.name}: split must be one of {VALID_SPLITS}")


@dataclass(frozen=True)
class Config:
    classes: ClassScheme
    recipe: Recipe
    regions: list[Region] = field(default_factory=list)

    def regions_in(self, *splits: str) -> list[Region]:
        return [r for r in self.regions if r.split in splits]


def load_classes(path: Path | None = None) -> ClassScheme:
    d = _load(path or CONFIG_DIR / "classes.yaml")
    classes = sorted(d["classes"], key=lambda c: c["id"])
    if [c["id"] for c in classes] != list(range(len(classes))):
        raise ValueError("class ids must be 0..N-1 without gaps")
    return ClassScheme(
        names=[c["name"] for c in classes],
        colors=[c["color"] for c in classes],
        ignore_index=int(d["ignore_index"]),
        worldcover_to_target={int(k): int(v) for k, v in d["worldcover_to_target"].items()},
        dynamicworld_to_target={int(k): int(v) for k, v in d["dynamicworld_to_target"].items()},
    )


def load_recipe(path: Path | None = None) -> Recipe:
    d = _load(path or CONFIG_DIR / "recipe.yaml")
    d["db_clip_percentiles"] = tuple(d["db_clip_percentiles"])
    return Recipe(**d)


def load_regions(path: Path | None = None) -> list[Region]:
    d = _load(path or CONFIG_DIR / "regions.yaml")
    return [Region(name=name, **vals) for name, vals in d["regions"].items()]


def load_config(config_dir: Path | None = None) -> Config:
    base = Path(config_dir) if config_dir else CONFIG_DIR
    return Config(
        classes=load_classes(base / "classes.yaml"),
        recipe=load_recipe(base / "recipe.yaml"),
        regions=load_regions(base / "regions.yaml"),
    )
