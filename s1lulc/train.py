"""Training loop.

Features: mixed precision on GPU, warm-up + cosine learning-rate schedule, separate encoder
learning rate for pretrained encoders, class-weighted CE + Dice loss, validation on full
regions with the production tiled predictor, early stopping on validation mIoU, best-model
bundle, resumable checkpoints (a disconnected session continues where it stopped), and a
per-epoch history for the training curves.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .analysis.metrics import confusion_matrix, mean_iou, overall_accuracy, summary
from .config import CONFIG_DIR
from .data.dataset import AugmentConfig, RandomPatchDataset, load_meta, load_regions
from .inference.bundle import BundleMeta, save_bundle
from .inference.predictor import predict_probs
from .models import build_model, count_parameters
from .preprocessing import NormStats, model_input

log = logging.getLogger(__name__)


def load_train_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "train.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by libraries
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, total: int, warmup: int, floor: float = 0.01) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def evaluate_regions(model, regions, stats: NormStats, n_classes: int, ignore: int,
                     months, device: str, tile: int, overlap: int,
                     batch_size: int = 8, temporal_context: int = 0) -> tuple[np.ndarray, float]:
    """Full-region prediction -> (confusion matrix ref x pred, mean NLL of the reference)."""
    cm = np.zeros((n_classes, n_classes), np.int64)
    nll_sum, nll_n = 0.0, 0
    for r in regions:
        for m in months:
            x, valid = model_input(r.db, m, stats, temporal_context)
            if not valid.any():
                continue
            p = predict_probs(model, x, n_classes, tile=tile, overlap=overlap,
                              batch_size=batch_size, device=device)
            y = r.label.astype(np.int64)
            keep = valid & (y != ignore)
            cm += confusion_matrix(y[keep], p.argmax(0)[keep], n_classes)
            pt = np.take_along_axis(p, np.where(keep, y, 0)[None], 0)[0][keep]
            nll_sum += float(-np.log(np.clip(pt, 1e-7, 1)).sum())
            nll_n += int(keep.sum())
    return cm, nll_sum / max(nll_n, 1)


def train_variant(data_dir: str | Path, out_dir: str | Path, variant: str,
                  cfg: dict | None = None, device: str | None = None,
                  resume: bool = True, epochs: int | None = None) -> Path:
    """Train one variant; returns the path of the best-model bundle."""
    import torch
    from segmentation_models_pytorch.losses import DiceLoss
    from torch.utils.data import DataLoader

    cfg = cfg or load_train_config()
    spec = cfg["variants"][variant]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or cfg["epochs"]
    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])

    data_dir = Path(data_dir)
    meta = load_meta(data_dir)
    stats = NormStats.load(data_dir / "norm.json")
    names, ignore = meta["class_names"], meta["ignore_index"]
    n_cls = len(names)
    train_regions = load_regions(data_dir, ("train",))
    val_regions = load_regions(data_dir, ("val",))
    log.info("%s: %d train / %d val regions on %s", variant, len(train_regions),
             len(val_regions), device)

    k = int(spec.get("temporal_context", 0))
    in_ch = len(meta["polarisations"]) * (2 * k + 1)
    ds = RandomPatchDataset(train_regions, stats, n_cls, ignore, cfg["patch_size"],
                            cfg["samples_per_epoch"], augment=AugmentConfig(**cfg["augment"]),
                            seed=cfg["seed"], temporal_context=k, **cfg["sampling"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=device.startswith("cuda"),
                        persistent_workers=False)
    log.info("rare classes (oversampled): %s", [names[c] for c in ds.rare_classes])

    model = build_model(spec["arch"], spec["encoder"], in_ch, n_cls,
                        spec.get("weights")).to(device)
    log.info("model %s/%s init=%s, input %d months (%d channels), %.1fM parameters",
             spec["arch"], spec["encoder"], spec.get("weights"), 2 * k + 1, in_ch,
             count_parameters(model) / 1e6)

    oc = cfg["optimizer"]
    enc_mult = oc["encoder_lr_mult"] if spec.get("weights") else 1.0
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": oc["lr"] * enc_mult},
                             {"params": other, "lr": oc["lr"]}],
                            weight_decay=oc["weight_decay"])
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, oc["warmup_epochs"] * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    weights = torch.tensor(meta["class_weights"], dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore)
    dice = DiceLoss("multiclass", ignore_index=ignore)
    lw = cfg["loss"]

    history, best, best_epoch, start_epoch = [], -1.0, 0, 0
    ckpt_path = out / "last.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best, best_epoch = ck["history"], ck["best"], ck["best_epoch"]
        start_epoch = ck["epoch"] + 1
        log.info("resumed from epoch %d (best val mIoU %.4f)", start_epoch, best)

    ic = cfg["inference"]
    val_months = [m - 1 for m in (cfg.get("val_months") or range(1, 13))]
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        ds.set_epoch(epoch)
        model.train()
        loss_sum, correct, counted = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(x)
                loss = lw["ce_weight"] * ce(logits.float(), y) + \
                    lw["dice_weight"] * dice(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss.detach())
            keep = y != ignore
            correct += int((logits.argmax(1)[keep] == y[keep]).sum())
            counted += int(keep.sum())

        cm, val_nll = evaluate_regions(model, val_regions, stats, n_cls, ignore, val_months,
                                       device, ic["tile"], ic["overlap"], temporal_context=k)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(steps_per_epoch, 1),
               "train_acc": correct / max(counted, 1), "val_nll": val_nll,
               "val_miou": mean_iou(cm), "val_oa": overall_accuracy(cm),
               "lr": opt.param_groups[1]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        improved = rec["val_miou"] > best
        if improved:
            best, best_epoch = rec["val_miou"], epoch + 1
            bmeta = BundleMeta(
                variant=variant, arch=spec["arch"], encoder=spec["encoder"],
                weights_init=spec.get("weights"), in_channels=in_ch, temporal_context=k,
                class_names=names, class_colors=meta["class_colors"], ignore_index=ignore,
                recipe_version=meta["recipe_version"], norm=stats.__dict__,
                label_year=meta["label_year"], best_epoch=best_epoch,
                val_metrics=summary(cm, names), confusion_val=cm.tolist())
            save_bundle(out / "bundle", model, bmeta)
        log.info("epoch %3d/%d  loss %.4f  acc %.3f | val mIoU %.4f OA %.3f NLL %.3f %s (%.0fs)",
                 epoch + 1, epochs, rec["train_loss"], rec["train_acc"], rec["val_miou"],
                 rec["val_oa"], val_nll, "*" if improved else " ", rec["seconds"])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best, "best_epoch": best_epoch,
                    "epoch": epoch}, ckpt_path)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if epoch + 1 - best_epoch >= cfg["early_stopping_patience"]:
            log.info("early stopping: no improvement for %d epochs",
                     cfg["early_stopping_patience"])
            break

    try:
        from .plots import plot_training_curves
        plot_training_curves(history, out / "training_curves.png",
                             f"{variant} (best val mIoU {best:.3f} at epoch {best_epoch})")
    except ImportError:
        pass
    (out / "train_config.json").write_text(json.dumps({"variant": variant, "spec": spec,
                                                       "config": cfg}, indent=2))
    return out / "bundle"
