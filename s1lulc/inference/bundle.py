"""Model bundle: everything needed to run a trained model correctly, in one folder.

    bundle/
      model.pt      state_dict
      bundle.json   architecture, classes, recipe version, normalisation statistics,
                    validation confusion matrix (for error-adjusted areas), noise floor,
                    evaluation summary

Loading checks that the bundle's recipe version equals the current recipe. A model trained
on differently prepared inputs is refused rather than silently giving wrong maps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..preprocessing import NormStats

BUNDLE_FORMAT = 1


class BundleMismatchError(RuntimeError):
    pass


@dataclass
class BundleMeta:
    variant: str
    arch: str
    encoder: str
    weights_init: str | None
    in_channels: int
    class_names: list[str]
    class_colors: list[str]
    ignore_index: int
    recipe_version: str
    norm: dict
    label_year: int
    best_epoch: int = 0
    temporal_context: int = 0          # k: input months t-k .. t+k
    val_metrics: dict = field(default_factory=dict)
    confusion_val: list[list[int]] = field(default_factory=list)   # rows ref, cols pred
    noise_floor: dict = field(default_factory=dict)
    test_summary: dict = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(
        timespec="seconds"))
    package_version: str = __version__
    bundle_format: int = BUNDLE_FORMAT

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def norm_stats(self) -> NormStats:
        return NormStats(**self.norm)


def save_bundle(path: str | Path, model, meta: BundleMeta) -> Path:
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    write_meta(path, meta)
    return path


def write_meta(path: str | Path, meta: BundleMeta) -> None:
    Path(path, "bundle.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def read_meta(path: str | Path) -> BundleMeta:
    d = json.loads(Path(path, "bundle.json").read_text(encoding="utf-8"))
    if d.get("bundle_format") != BUNDLE_FORMAT:
        raise BundleMismatchError(f"unsupported bundle format {d.get('bundle_format')}")
    return BundleMeta(**d)


def load_bundle(path: str | Path, device: str = "cpu",
                expected_recipe_version: str | None = None):
    """Return (model in eval mode on ``device``, BundleMeta)."""
    import torch

    from ..models import build_model

    meta = read_meta(path)
    if expected_recipe_version and meta.recipe_version != expected_recipe_version:
        raise BundleMismatchError(
            f"model was trained on recipe {meta.recipe_version}, current recipe is "
            f"{expected_recipe_version}; inputs would be prepared differently")
    model = build_model(meta.arch, meta.encoder, meta.in_channels, meta.n_classes,
                        weights=None)
    state = torch.load(Path(path, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), meta
