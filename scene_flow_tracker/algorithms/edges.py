from __future__ import annotations

import cv2
import numpy as np


def grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def valid_image_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = grayscale(image_rgb)
    return gray > 3


def edge_map(image_rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    gray = grayscale(image_rgb)
    threshold1 = int(cfg.get("threshold1", 50))
    threshold2 = int(cfg.get("threshold2", 150))
    edges = cv2.Canny(gray, threshold1, threshold2)
    if bool(cfg.get("closing_enabled", False)):
        k = max(1, int(cfg.get("closing_kernel_size", 3)))
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    strength = cv2.Laplacian(gray, cv2.CV_32F)
    strength = np.abs(strength)
    if strength.max() > 0:
        strength = strength / strength.max()
    return edges > 0, strength.astype(np.float32)


def filter_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == idx] = True
    return out


def bbox_mask(shape: tuple[int, int], bbox: np.ndarray | None, expand_ratio: float = 0.0) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if bbox is None:
        return mask
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= bw * expand_ratio
    x2 += bw * expand_ratio
    y1 -= bh * expand_ratio
    y2 += bh * expand_ratio
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(w - 1, int(np.ceil(x2)))
    iy2 = min(h - 1, int(np.ceil(y2)))
    if ix2 >= ix1 and iy2 >= iy1:
        mask[iy1 : iy2 + 1, ix1 : ix2 + 1] = True
    return mask
