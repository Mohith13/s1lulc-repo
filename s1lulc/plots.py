"""Report figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_class_distribution(split_counts: dict[str, np.ndarray], names: list[str],
                            path: Path) -> None:
    plt = _plt()
    splits = list(split_counts)
    x = np.arange(len(names))
    w = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, s in enumerate(splits):
        c = np.asarray(split_counts[s], float)
        ax.bar(x + i * w - 0.4 + w / 2, 100 * c / max(c.sum(), 1), w, label=s)
    ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=20)
    ax.set_ylabel("share of labelled pixels (%)")
    ax.set_title("Class distribution by split (ESA WorldCover 2021, merged classes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], path: Path, title: str = "") -> None:
    plt = _plt()
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train (CE + Dice)")
    ax[0].plot(ep, [h["val_nll"] for h in history], label="validation (NLL)")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_miou"] for h in history], label="val mIoU")
    ax[1].plot(ep, [h["val_oa"] for h in history], label="val OA")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train pixel acc.", alpha=0.6)
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[2].plot(ep, [h["lr"] for h in history])
    ax[2].set_title("Learning rate (decoder)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, names: list[str], path: Path, title: str = "") -> None:
    """Row-normalised confusion matrix (rows = reference, values = recall)."""
    plt = _plt()
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = [n.replace("_", "\n") for n in names]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), labels, fontsize=8)
    ax.set_yticks(range(len(names)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference (WorldCover)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_metric(per_month: dict[str, list[float]], path: Path, ylabel: str,
                        title: str = "") -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
              "Dec"]
    for k, v in per_month.items():
        ax.plot(months, v, marker="o", label=k)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(bin_conf: np.ndarray, bin_acc: np.ndarray, bin_count: np.ndarray,
                     ece: float, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ok = bin_count > 0
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(bin_conf[ok], bin_acc[ok], marker="o", label=f"model (ECE = {ece:.3f})")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
