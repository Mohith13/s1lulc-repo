"""Sliding-window inference over arbitrarily sized images.

The image is covered by overlapping tiles; each tile's softmax output is multiplied by a
smooth weight window (high in the centre, low at the edges) and the weighted sum is
normalised. This removes visible tile seams and the poorer predictions CNNs make near tile
borders. Evaluation and the prototype use exactly this function.
"""

from __future__ import annotations

import numpy as np

from ..preprocessing import NormStats, model_input


def tile_positions(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, size) with tiles of length ``tile``; last tile flush."""
    if size <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def weight_window(tile: int, floor: float = 0.05) -> np.ndarray:
    w = np.hanning(tile + 2)[1:-1]
    w = np.maximum(np.outer(w, w), floor)
    return (w / w.max()).astype(np.float32)


def predict_probs(model, x: np.ndarray, n_classes: int, tile: int = 256, overlap: int = 64,
                  batch_size: int = 8, device: str = "cpu") -> np.ndarray:
    """x: (C_in, H, W) normalised input -> (n_classes, H, W) float32 probabilities."""
    import torch

    c, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:                                  # images smaller than a tile: reflect-pad
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    H, W = x.shape[1:]
    win = weight_window(tile)
    acc = np.zeros((n_classes, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    coords = [(y, xx) for y in tile_positions(H, tile, overlap)
              for xx in tile_positions(W, tile, overlap)]

    model.eval()
    use_amp = device.startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([x[:, y:y + tile, xx:xx + tile] for y, xx in chunk])
            inp = torch.from_numpy(batch).to(device)
            with torch.autocast("cuda", enabled=use_amp):
                prob = torch.softmax(model(inp).float(), dim=1).cpu().numpy()
            for (y, xx), p in zip(chunk, prob, strict=True):
                acc[:, y:y + tile, xx:xx + tile] += p * win
                wsum[y:y + tile, xx:xx + tile] += win
    probs = acc / wsum
    return probs[:, :h, :w]


def predict_series(model, db: np.ndarray, stats: NormStats, n_classes: int,
                   device: str = "cpu", temporal_context: int = 0,
                   **kw) -> tuple[np.ndarray, np.ndarray]:
    """db: (T, 2, H, W) dB composites -> probs (T, C, H, W) float16, valid (T, H, W)."""
    t, _, h, w = db.shape
    probs = np.zeros((t, n_classes, h, w), np.float16)
    valid = np.zeros((t, h, w), bool)
    for i in range(t):
        x, v = model_input(db, i, stats, temporal_context)
        valid[i] = v
        if v.any():
            probs[i] = predict_probs(model, x, n_classes, device=device, **kw)
    return probs, valid
