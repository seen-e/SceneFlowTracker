from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..jobs import SegmentJob


def _view_name(view_key: str) -> str:
    return str(view_key).split(".")[-1]


def _depth_path(root: str | Path, job: SegmentJob) -> Path | None:
    base = Path(root)
    view = _view_name(job.view_key)
    candidates = [
        base / job.episode_id / f"{view}.npz",
        base / job.dataset / job.episode_id / f"{view}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=16)
def _load_depth_npz(path: str) -> tuple[np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as data:
        depths = np.asarray(data["depths"], dtype=np.float32).copy()
        fps = float(data["fps"]) if "fps" in data.files else 1.0
        start_sec = float(data["start_sec"]) if "start_sec" in data.files else 0.0
    return depths, fps, start_sec


def _normalize_depth(depth: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    values = depth[valid]
    if values.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    low, high = cfg.get("normalize_percentiles", [2.0, 98.0])
    lo, hi = np.percentile(values, [float(low), float(high)])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    norm[~valid] = 0.0
    blur_ksize = int(cfg.get("blur_ksize", 3))
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        norm = cv2.GaussianBlur(norm, (blur_ksize, blur_ksize), 0)
    return norm


def depth_edge_mask(depth: np.ndarray, image_shape: tuple[int, int], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image_shape
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool), {"status": "invalid_depth"}

    fill = float(np.nanmedian(depth[valid]))
    depth = np.where(valid, depth, fill).astype(np.float32)
    norm = _normalize_depth(depth, valid, cfg)
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=int(cfg.get("sobel_ksize", 3)))
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=int(cfg.get("sobel_ksize", 3)))
    grad = np.sqrt(gx * gx + gy * gy)
    grad_values = grad[valid]
    percentile = float(cfg.get("gradient_percentile", 85.0))
    threshold = float(np.percentile(grad_values, percentile)) if grad_values.size else 1.0
    threshold = max(threshold, float(cfg.get("min_gradient", 0.02)))
    mask = (grad >= threshold) & valid

    dilate_px = int(cfg.get("dilate_px", 1))
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask, {
        "status": "ok",
        "edge_pixels": int(mask.sum()),
        "gradient_threshold": float(threshold),
        "gradient_percentile": percentile,
    }


def depth_edge_for_segment(job: SegmentJob, cfg: dict[str, Any], image_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not bool((cfg or {}).get("enabled", False)):
        return None, {"enabled": False, "status": "disabled"}
    fallback = bool(cfg.get("fallback_to_rgb_when_missing", True))
    empty = np.zeros(image_shape, dtype=bool)
    root = cfg.get("root")
    if not root:
        stats = {"enabled": True, "status": "missing_root", "fallback_to_rgb": fallback}
        return (None if fallback else empty), stats
    path = _depth_path(root, job)
    if path is None:
        stats = {"enabled": True, "status": "missing_depth_npz", "root": str(root), "view": _view_name(job.view_key), "fallback_to_rgb": fallback}
        return (None if fallback else empty), stats
    try:
        depths, depth_fps, start_sec = _load_depth_npz(str(path))
        source_fps = float(job.manifest_fps or job.effective_fps or 0.0)
        if source_fps <= 0:
            stats = {"enabled": True, "status": "invalid_source_fps", "path": str(path), "fallback_to_rgb": fallback}
            return (None if fallback else empty), stats
        segment_sec = float(job.episode_start_frame) / source_fps
        depth_index = int(round((segment_sec - start_sec) * depth_fps))
        depth_index = max(0, min(int(depths.shape[0]) - 1, depth_index))
        mask, stats = depth_edge_mask(depths[depth_index], image_shape, cfg)
        stats.update(
            {
                "enabled": True,
                "path": str(path),
                "depth_fps": float(depth_fps),
                "depth_frame_index": int(depth_index),
                "segment_start_sec": float(segment_sec),
            }
        )
        if stats.get("status") != "ok":
            stats["fallback_to_rgb"] = fallback
            return (None if fallback else empty), stats
        return mask, stats
    except Exception as exc:
        stats = {"enabled": True, "status": "error", "path": str(path), "error": str(exc), "fallback_to_rgb": fallback}
        return (None if fallback else empty), stats
