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
    if excluded and len(picked):
        picked = np.asarray([p for p in picked if (int(round(float(p[0]))), int(round(float(p[1])))) not in excluded], dtype=np.float32).reshape(-1, 2)
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
    if len(picked) < target:
        raise RuntimeError("INSUFFICIENT_UNIQUE_QUERY_POINTS")
    picked = picked[:target].astype(np.float32)
    pix = np.round(picked).astype(int)
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
            "candidate_count": int(len(points)),
            "valid_region_bbox_xyxy": [int(v) for v in valid_bbox],
            "edge_constraint_used": bool(used_edge_constraint),
            "constraint_fallback_level": constraint_fallback_level,
        },
    }
