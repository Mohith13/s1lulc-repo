"""Write prototype results: GeoTIFFs, CSV tables, JSON summary and figures."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from ..geo import PixelGrid


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def label_rgba(labels: np.ndarray, colors: list[str], ignore: int = 255) -> np.ndarray:
    lut = np.zeros((256, 4), np.uint8)
    for i, c in enumerate(colors):
        lut[i, :3], lut[i, 3] = hex_to_rgb(c), 255
    lut[ignore] = 0
    return lut[labels]


def gray_rgba(db: np.ndarray, vmin: float = -25, vmax: float = 5) -> np.ndarray:
    g = np.clip((np.nan_to_num(db, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    g = (g * 255).astype(np.uint8)
    a = np.where(np.isfinite(db), 255, 0).astype(np.uint8)
    return np.dstack([g, g, g, a])


def png_bytes(rgba: np.ndarray, max_size: int = 500) -> bytes:
    import matplotlib.image as mpimg

    step = max(1, int(np.ceil(max(rgba.shape[:2]) / max_size)))
    buf = io.BytesIO()
    mpimg.imsave(buf, rgba[::step, ::step], format="png")
    return buf.getvalue()


# --- files -----------------------------------------------------------------------------------
def write_geotiff(path: Path, data: np.ndarray, grid: PixelGrid, nodata=None,
                  band_names: list[str] | None = None) -> None:
    import rasterio
    from rasterio.transform import Affine

    data = data if data.ndim == 3 else data[None]
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height,
                       count=data.shape[0], dtype=data.dtype, crs=grid.crs, nodata=nodata,
                       transform=Affine(*grid.transform), compress="deflate") as dst:
        dst.write(data)
        for i, name in enumerate(band_names or [], start=1):
            dst.set_band_description(i, name)


def _csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _r(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def write_tables(out: Path, res: dict, months: list[str]) -> None:
    names = res["class_names"]
    raw, sm = res["series"]["raw"], res["series"]["smoothed"]
    dw = res.get("dynamic_world")
    rows = []
    for i, m in enumerate(months):
        for c, name in enumerate(names):
            rows.append([m, name, _r(raw["shares"][i, c]), _r(sm["shares"][i, c]),
                         _r(raw["adjusted"][i, c]) if "adjusted" in raw else "",
                         _r(raw["adjusted_ci95"][i, c]) if "adjusted" in raw else "",
                         _r(dw["shares"][i, c]) if dw else ""])
    _csv(out / "class_shares.csv", ["month", "class", "share_pct", "share_smoothed_pct",
                                    "share_adjusted_pct", "adjusted_ci95_pct",
                                    "dynamic_world_pct"], rows)
    rows = []
    for i in range(len(months) - 1):
        for c, name in enumerate(names):
            rows.append([months[i], months[i + 1], name, _r(raw["month_to_month"][i, c]),
                         bool(raw["significant_m2m"][i, c]) if "significant_m2m" in raw else "",
                         _r(sm["month_to_month"][i, c]),
                         bool(sm["significant_m2m"][i, c]) if "significant_m2m" in sm else ""])
    _csv(out / "change_month_to_month.csv", ["from", "to", "class", "change_pp", "significant",
                                             "change_smoothed_pp", "significant_smoothed"], rows)
    rows = [[name, _r(raw["shares"][raw["first_month"], c]),
             _r(raw["shares"][raw["last_month"], c]), _r(raw["total"][c]),
             bool(raw["significant_total"][c]) if "significant_total" in raw else "",
             _r(raw["floor_total"][c]) if "floor_total" in raw else "",
             _r(sm["total"][c]), _r(raw["gross_gain"][c]), _r(raw["gross_loss"][c])]
            for c, name in enumerate(names)]
    _csv(out / "change_total.csv", ["class", "first_month_pct", "last_month_pct", "change_pp",
                                    "significant", "noise_floor_pp", "change_smoothed_pp",
                                    "gross_gain_pp", "gross_loss_pp"], rows)
    _csv(out / "transitions_first_to_last.csv", ["from \\ to", *names],
         [[names[i], *[_r(v) for v in raw["transitions"][i]]] for i in range(len(names))])
    if dw:
        _csv(out / "dynamic_world_agreement.csv", ["month", "agreement", "coverage"],
             [[m, _r(dw["agreement"][i]), _r(dw["coverage"][i])] for i, m in enumerate(months)])


def platform_label(platforms: list[str]) -> str:
    return " + ".join(f"S1{p}" for p in platforms) if platforms else "none"


def write_acquisitions(path: Path, res: dict, months: list[str], dw_images=None) -> None:
    pm = res.get("passes_median") or [None] * len(months)
    sc = res.get("scenes") or [None] * len(months)
    _csv(path, ["month", "scenes", "median_passes_per_pixel", "satellites",
                "dynamic_world_images"],
         [[m, sc[i], pm[i], platform_label(res["platforms"][i]),
           dw_images[i] if dw_images else ""] for i, m in enumerate(months)])


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items() if k != "maps"}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


# --- figures ---------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    return buf.getvalue()


def figure_map_grid(maps: np.ndarray, colors, names, months, title: str, ignore=255):
    plt = _plt()
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(label_rgba(maps[i], colors, ignore), interpolation="nearest")
        ax.set_title(months[i], fontsize=10)
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=n.replace("_", " ")) for c, n in
                        zip(colors, names, strict=True)],
               loc="lower center", ncol=len(names), fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def figure_shares(res: dict, colors, months):
    plt = _plt()
    names = res["class_names"]
    raw = res["series"]["raw"]
    x = np.arange(len(months))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].stackplot(x, np.nan_to_num(raw["shares"]).T, colors=colors,
                      labels=[n.replace("_", " ") for n in names])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("share of area (%)")
    axes[0].set_title("Class shares per month (model, raw)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    for c, name in enumerate(names):
        axes[1].plot(x, raw["shares"][:, c], color=colors[c], lw=1.2, alpha=0.5)
        axes[1].plot(x, res["series"]["smoothed"]["shares"][:, c], color=colors[c], lw=2,
                     label=name.replace("_", " "))
        if "adjusted" in raw:
            a, ci = raw["adjusted"][:, c], raw["adjusted_ci95"][:, c]
            axes[1].fill_between(x, a - ci, a + ci, color=colors[c], alpha=0.12, lw=0)
    axes[1].set_title("Raw (thin), smoothed (thick); band = error-adjusted 95% CI")
    axes[1].set_ylabel("share of area (%)")
    for ax in axes:
        ax.set_xticks(x, months, rotation=45, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figure_changes(res: dict, months):
    plt = _plt()
    names = [n.replace("_", " ") for n in res["class_names"]]
    labels = [f"{a}→{b}" for a, b in zip(months[:-1], months[1:], strict=True)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
    for ax, kind in zip(axes, ("raw", "smoothed"), strict=True):
        s = res["series"][kind]
        d = s["month_to_month"].T
        lim = max(1.0, float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0)
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        sig = s.get("significant_m2m")
        for c in range(d.shape[0]):
            for i in range(d.shape[1]):
                if np.isfinite(d[c, i]):
                    star = "*" if sig is not None and sig[i, c] else ""
                    ax.text(i, c, f"{d[c, i]:+.1f}{star}", ha="center", va="center",
                            fontsize=7)
        ax.set_xticks(range(len(labels)), labels, rotation=60, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"Month-to-month change, pp ({kind}); * = above noise floor")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def figure_uncertainty(res: dict, colors):
    plt = _plt()
    raw = res["series"]["raw"]
    names = [n.replace("_", " ") for n in res["class_names"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    im = axes[0].imshow(res["maps"]["entropy_mean"], cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("Mean uncertainty (normalised entropy)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    freq = np.ma.masked_less(res["maps"]["raw_change_frequency"], 0)
    im = axes[1].imshow(freq, cmap="viridis", vmin=0, vmax=max(3, int(freq.max() or 1)))
    axes[1].set_title("Label switches per pixel over the year")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for a in axes[:2]:
        a.axis("off")
    y = np.arange(len(names))
    axes[2].barh(y - 0.2, raw["gross_gain"], 0.4, color="#2b8cbe", label="gross gain")
    axes[2].barh(y + 0.2, -raw["gross_loss"], 0.4, color="#e34a33", label="gross loss")
    axes[2].scatter(raw["total"], y, color="k", zorder=3, label="net change")
    axes[2].set_yticks(y, names, fontsize=8)
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_xlabel("percentage points (first → last month)")
    axes[2].set_title(f"Gross vs net change ({raw['changed_total']:.1f}% of area changed label)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def figure_dw(res: dict, months):
    plt = _plt()
    dw = res["dynamic_world"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [np.nan if v is None else 100 * v for v in dw["agreement"]]
    ax.plot(months, vals, marker="o", label="pixel agreement with Dynamic World")
    ax.plot(months, [100 * c for c in dw["coverage"]], ls="--", color="gray",
            label="Dynamic World coverage (cloud-free)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Independent check: agreement with Dynamic World (Sentinel-2 based)")
    fig.tight_layout()
    return fig
