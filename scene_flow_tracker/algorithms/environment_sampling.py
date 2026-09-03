from __future__ import annotations

import numpy as np

from .edges import bbox_mask, content_region_mask, edge_map, filter_small_components
from .robot_sampling import _spatial_pick
from .trackability import trackability_map


def _grid_points(mask: np.ndarray, target: int) -> np.ndarray:
    h, w = mask.shape
    if target <= 0:
        return np.empty((0, 2), np.float32)
    side = int(np.ceil(np.sqrt(target)))
    xs = np.linspace(0, w - 1, max(1, side * 2)).round().astype(int)
    ys = np.linspace(0, h - 1, max(1, side * 2)).round().astype(int)
    pts = []
    for y in ys:
        for x in xs:
            if mask[y, x]:
                pts.append((x, y))
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def _repeat_with_jitter(points: np.ndarray, target: int, width: int, height: int, rng: np.random.Generator, jitter_px: float) -> np.ndarray:
    if len(points) <= 0 or len(points) >= target:
        return points
    needed = int(target) - len(points)
    source_indices = rng.integers(0, len(points), size=needed)
    repeated = points[source_indices].astype(np.float32, copy=True)
    if jitter_px > 0:
        jitter = rng.uniform(-jitter_px, jitter_px, size=repeated.shape).astype(np.float32)
        repeated += jitter
    repeated[:, 0] = np.clip(repeated[:, 0], 0, max(0, width - 1))
    repeated[:, 1] = np.clip(repeated[:, 1], 0, max(0, height - 1))
    return np.concatenate([points, repeated], axis=0)


def _seed_points_for_repeat(points: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(points) > 0:
        return points.astype(np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) <= 0:
        return np.empty((0, 2), np.float32)
    idx = int(rng.integers(0, len(xs)))
    return np.asarray([[xs[idx], ys[idx]]], dtype=np.float32)


def sample_environment_points(
    image_rgb: np.ndarray,
    robot_bboxes: list[np.ndarray],
    target: int,
    cfg: dict,
    seed: int = 0,
    exclude_points: np.ndarray | None = None,
    edge_constraint_mask: np.ndarray | None = None,
) -> dict:
    h, w = image_rgb.shape[:2]
    if target <= 0:
        return {"points_xy": np.empty((0, 2), np.float32), "features": {}, "stats": {"target_points": int(target), "final_sampled_count": 0}}
    content_valid, valid_bbox = content_region_mask(image_rgb, cfg.get("valid_region", {}))
    valid = content_valid.copy()
    margin = float(cfg.get("robot_exclusion_margin_px", 0))
    for bbox in robot_bboxes:
        if bbox is not None:
            expand = margin / max(1.0, max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])))
            valid &= ~bbox_mask((h, w), bbox, expand)
    excluded = set()
    if exclude_points is not None:
        excluded = {(int(round(x)), int(round(y))) for x, y in np.asarray(exclude_points).reshape(-1, 2)}
    edges, edge_strength = edge_map(image_rgb, cfg.get("edge", {}))
    edges = filter_small_components(edges, int(cfg.get("edge", {}).get("min_component_pixels", 5)))
    track = trackability_map(image_rgb, cfg.get("trackability", {})) if cfg.get("trackability", {}).get("enabled", True) else np.zeros((h, w), np.float32)
    constraint_fallback_level = "none"
    if edge_constraint_mask is not None:
        constraint = np.asarray(edge_constraint_mask, dtype=bool)
        if constraint.shape != (h, w):
            raise ValueError(f"edge_constraint_mask expects {(h, w)}, got {constraint.shape}")
        candidate = valid & edges & constraint
        fallback_valid = candidate
        used_edge_constraint = True
    else:
        candidate = valid & (edges | (track > np.percentile(track[valid], 80) if np.any(valid) else False))
        fallback_valid = valid
        used_edge_constraint = False
    ys, xs = np.nonzero(candidate)
    points = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.empty((0, 2), np.float32)
    score = edge_strength[ys, xs] + track[ys, xs] if len(xs) else np.empty((0,), np.float32)
    rng = np.random.default_rng(seed)
    picked = _spatial_pick(points, score, int(target), (h, w), cfg.get("spatial_sampling", {}), rng)
    unique_sampled_count = int(len(picked))
    if excluded and len(picked):
        picked = np.asarray([p for p in picked if (int(round(float(p[0]))), int(round(float(p[1])))) not in excluded], dtype=np.float32).reshape(-1, 2)
        unique_sampled_count = int(len(picked))
    used = set(excluded)
    used.update((int(round(x)), int(round(y))) for x, y in picked)
    if len(picked) < target:
        if used_edge_constraint:
            fallback_valid = valid & edges
            constraint_fallback_level = "rgb_edge"
        fallback = _grid_points(fallback_valid, target * 4)
        rng.shuffle(fallback)
        extra = []
        for x, y in fallback:
            key = (int(round(x)), int(round(y)))
            if key in used:
                continue
            extra.append([x, y])
            used.add(key)
            if len(picked) + len(extra) >= target:
                break
        if extra:
            picked = np.concatenate([picked, np.asarray(extra, dtype=np.float32)], axis=0)
    if len(picked) < target:
        if used_edge_constraint and constraint_fallback_level == "rgb_edge":
            fallback_valid = valid
            constraint_fallback_level = "roi"
        ys, xs = np.nonzero(fallback_valid if used_edge_constraint else content_valid)
        order = np.arange(len(xs))
        rng.shuffle(order)
        extra = []
        for idx in order:
            key = (int(xs[idx]), int(ys[idx]))
            if key in used:
                continue
            extra.append([xs[idx], ys[idx]])
            used.add(key)
            if len(picked) + len(extra) >= target:
                break
        if extra:
            picked = np.concatenate([picked, np.asarray(extra, dtype=np.float32)], axis=0)
    repeat_padding_count = 0
    repeat_jitter_px = float(cfg.get("repeat_jitter_px", 1.0))
    unique_sampled_count = int(min(len(picked), target))
    if len(picked) < target:
        unique_sampled_count = int(len(picked))
        seed_points = _seed_points_for_repeat(picked, fallback_valid if used_edge_constraint else content_valid, rng)
        if len(seed_points) <= 0:
            seed_points = _seed_points_for_repeat(points, content_valid, rng)
        if len(seed_points) <= 0:
            seed_points = np.asarray([[(valid_bbox[0] + valid_bbox[2]) * 0.5, (valid_bbox[1] + valid_bbox[3]) * 0.5]], dtype=np.float32)
        before = len(seed_points)
        unique_sampled_count = max(unique_sampled_count, int(before))
        picked = _repeat_with_jitter(seed_points.astype(np.float32), int(target), w, h, rng, repeat_jitter_px)
        repeat_padding_count = int(len(picked) - before)
        constraint_fallback_level = "repeat_jitter"
    picked = picked[:target].astype(np.float32)
    pix = np.round(picked).astype(int)
    pix[:, 0] = np.clip(pix[:, 0], 0, max(0, w - 1))
    pix[:, 1] = np.clip(pix[:, 1], 0, max(0, h - 1))
    features = {
        "edge_strength": edge_strength[pix[:, 1], pix[:, 0]].astype(np.float32),
        "trackability_score": track[pix[:, 1], pix[:, 0]].astype(np.float32),
    }
    features["sampling_score"] = (features["edge_strength"] + features["trackability_score"]).astype(np.float32)
    return {
        "points_xy": picked,
        "features": features,
        "stats": {
            "target_points": int(target),
            "final_sampled_count": int(len(picked)),
            "unique_sampled_count": int(unique_sampled_count),
            "repeat_padding_count": int(repeat_padding_count),
            "repeat_jitter_px": float(repeat_jitter_px),
            "candidate_count": int(len(points)),
            "valid_region_bbox_xyxy": [int(v) for v in valid_bbox],
            "edge_constraint_used": bool(used_edge_constraint),
            "constraint_fallback_level": constraint_fallback_level,
        },
    }
