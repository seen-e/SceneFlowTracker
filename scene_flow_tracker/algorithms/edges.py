from __future__ import annotations

import cv2
import numpy as np

DEFAULT_VALID_REGION_CFG = {
    "enabled": True,
    "black_threshold": 5,
    "min_row_content_fraction": 0.02,
    "min_col_content_fraction": 0.02,
    "inset_px": 4,
    "min_area_fraction": 0.25,
}


def grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def valid_image_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = grayscale(image_rgb)
    return gray > 3


def _valid_region_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_VALID_REGION_CFG)
    out.update(cfg or {})
    return out


def detect_content_bbox(image_rgb: np.ndarray, cfg: dict | None = None) -> tuple[int, int, int, int]:
    region_cfg = _valid_region_cfg(cfg)
    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0 or not bool(region_cfg.get("enabled", True)):
        return 0, 0, max(0, w - 1), max(0, h - 1)
    if region_cfg.get("bbox_xyxy") is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in region_cfg["bbox_xyxy"]]
        x1 = min(w - 1, max(0, x1))
        x2 = min(w - 1, max(0, x2))
        y1 = min(h - 1, max(0, y1))
        y2 = min(h - 1, max(0, y2))
        if x2 >= x1 and y2 >= y1:
            return x1, y1, x2, y2

    gray = grayscale(image_rgb)
    non_black = gray > int(region_cfg.get("black_threshold", 5))
    row_frac = non_black.mean(axis=1)
    col_frac = non_black.mean(axis=0)
    rows = np.flatnonzero(row_frac >= float(region_cfg.get("min_row_content_fraction", 0.02)))
    cols = np.flatnonzero(col_frac >= float(region_cfg.get("min_col_content_fraction", 0.02)))
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w - 1, h - 1

    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    area_fraction = ((x2 - x1 + 1) * (y2 - y1 + 1)) / max(1, h * w)
    if area_fraction < float(region_cfg.get("min_area_fraction", 0.25)):
        return 0, 0, w - 1, h - 1

    inset = max(0, int(region_cfg.get("inset_px", 4)))
    x1 = min(w - 1, max(0, x1 + inset))
    x2 = min(w - 1, max(0, x2 - inset))
    y1 = min(h - 1, max(0, y1 + inset))
    y2 = min(h - 1, max(0, y2 - inset))
    if x2 < x1 or y2 < y1:
        return 0, 0, w - 1, h - 1
    return x1, y1, x2, y2


def content_region_mask(image_rgb: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = detect_content_bbox(image_rgb, cfg)
    mask = np.zeros((h, w), dtype=bool)
    if x2 >= x1 and y2 >= y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask, (x1, y1, x2, y2)


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
