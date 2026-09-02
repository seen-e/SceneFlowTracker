from __future__ import annotations

import cv2
import numpy as np

from .edges import bbox_mask, content_region_mask, edge_map, filter_small_components
from .trackability import trackability_map


def _color_score(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    a = np.abs(lab[:, :, 1] - 128.0)
    b = np.abs(lab[:, :, 2] - 128.0)
    neutral = np.clip(1.0 - (a + b) / 80.0, 0.0, 1.0)
    black = np.clip((95.0 - l) / 95.0, 0.0, 1.0)
    white = np.clip((l - 155.0) / 100.0, 0.0, 1.0)
    return np.maximum(black, white) * neutral


def _spatial_pick(points: np.ndarray, scores: np.ndarray, target: int, image_shape: tuple[int, int], cfg: dict, rng: np.random.Generator) -> np.ndarray:
    if target <= 0 or len(points) == 0:
        return np.empty((0, 2), np.float32)
    h, w = image_shape
    rows = max(1, int(cfg.get("grid_rows", 5)))
    cols = max(1, int(cfg.get("grid_cols", 5)))
    min_dist = float(cfg.get("min_point_distance", 5))
    max_per_cell = max(1, int(cfg.get("max_points_per_cell", max(1, target))))
    max_candidates = int(cfg.get("max_candidate_points", max(5000, target * 30)))
    if len(points) > max_candidates:
        top_idx = np.argpartition(-scores, max_candidates - 1)[:max_candidates]
        points = points[top_idx]
        scores = scores[top_idx]
    order = np.argsort(-scores, kind="mergesort")
    picked_arr = np.empty((int(target), 2), dtype=np.float32)
    picked_count = 0
    per_cell: dict[tuple[int, int], int] = {}
    for idx in order:
        p = points[idx]
        c = min(cols - 1, max(0, int(p[0] / max(1, w) * cols)))
        r = min(rows - 1, max(0, int(p[1] / max(1, h) * rows)))
        key = (r, c)
        if per_cell.get(key, 0) >= max_per_cell and picked_count < target:
            continue
        if picked_count:
            dist2 = np.sum((picked_arr[:picked_count] - p) ** 2, axis=1)
            if np.any(dist2 < min_dist * min_dist):
                continue
        picked_arr[picked_count] = p
        picked_count += 1
        per_cell[key] = per_cell.get(key, 0) + 1
        if picked_count >= target:
            break
    if picked_count < target:
        used = {(int(round(float(q[0]))), int(round(float(q[1])))) for q in picked_arr[:picked_count]}
        remaining = [points[idx] for idx in order if (int(round(float(points[idx][0]))), int(round(float(points[idx][1])))) not in used]
        if remaining:
            remaining_arr = np.asarray(remaining, dtype=np.float32)
            rng.shuffle(remaining_arr)
        else:
            remaining_arr = np.empty((0, 2), np.float32)
        for p in remaining_arr:
            picked_arr[picked_count] = p
            picked_count += 1
            if picked_count >= target:
                break
    return picked_arr[:picked_count].copy()


def sample_robot_points(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray | None,
    target: int,
    cfg: dict,
    seed: int = 0,
    edge_constraint_mask: np.ndarray | None = None,
) -> dict:
    if bbox_xyxy is None or target <= 0:
        return {"points_xy": np.empty((0, 2), np.float32), "features": {}, "stats": {"target_points": int(target), "final_sampled_count": 0}}
    h, w = image_rgb.shape[:2]
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    valid_region, valid_bbox = content_region_mask(image_rgb, cfg.get("valid_region", {}))
    mask = bbox_mask((h, w), bbox, float(cfg.get("topology", {}).get("bbox_expand_ratio", 0.0))) & valid_region
    edges, edge_strength = edge_map(image_rgb, cfg.get("edge", {}))
    edges = filter_small_components(edges, int(cfg.get("edge", {}).get("min_component_pixels", 5)))
    track = trackability_map(image_rgb, cfg.get("trackability", {})) if cfg.get("trackability", {}).get("enabled", True) else np.zeros((h, w), np.float32)
    color = _color_score(image_rgb) if cfg.get("color", {}).get("enabled", True) else np.zeros((h, w), np.float32)
    constraint_fallback_level = "none"
    if edge_constraint_mask is not None:
        constraint = np.asarray(edge_constraint_mask, dtype=bool)
        if constraint.shape != (h, w):
            raise ValueError(f"edge_constraint_mask expects {(h, w)}, got {constraint.shape}")
        candidate = mask & edges & constraint
        used_edge_constraint = True
    else:
        candidate = mask & (edges | (track > np.percentile(track, 85)))
        used_edge_constraint = False
    ys, xs = np.nonzero(candidate)
    if len(xs) == 0 and not used_edge_constraint:
        ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.empty((0, 2), np.float32)
    score = edge_strength[ys, xs] + track[ys, xs] + color[ys, xs] if len(xs) else np.empty((0,), np.float32)
    rng = np.random.default_rng(seed)
    picked = _spatial_pick(points, score, int(target), (h, w), cfg.get("spatial_sampling", {}), rng)
    if used_edge_constraint and len(picked) < target:
        used = {(int(round(float(x))), int(round(float(y)))) for x, y in picked}
        fb_mask = mask & edges
        ys_fb, xs_fb = np.nonzero(fb_mask)
        fb_points = np.stack([xs_fb, ys_fb], axis=1).astype(np.float32) if len(xs_fb) else np.empty((0, 2), np.float32)
        fb_score = (edge_strength[ys_fb, xs_fb] + track[ys_fb, xs_fb] + color[ys_fb, xs_fb]) if len(xs_fb) else np.empty((0,), np.float32)
        fb = _spatial_pick(fb_points, fb_score.astype(np.float32), int(target) - len(picked), (h, w), cfg.get("spatial_sampling", {}), rng)
        if len(fb):
            extras = []
            for x, y in fb:
                key = (int(round(float(x))), int(round(float(y))))
                if key in used:
                    continue
                extras.append([x, y])
                used.add(key)
            if extras:
                picked = np.concatenate([picked, np.asarray(extras, dtype=np.float32)], axis=0)
                constraint_fallback_level = "rgb_edge"
    pix = np.round(picked).astype(int) if len(picked) else np.empty((0, 2), int)
    features = {
        "sampling_score": score[:0].astype(np.float32),
        "edge_strength": np.empty((0,), np.float32),
        "trackability_score": np.empty((0,), np.float32),
        "color_score": np.empty((0,), np.float32),
        "topology_score": np.empty((0,), np.float32),
        "candidate_level": np.asarray(["robot_candidate"] * len(picked)),
    }
    if len(picked):
        px, py = pix[:, 0], pix[:, 1]
        features.update({
            "edge_strength": edge_strength[py, px].astype(np.float32),
            "trackability_score": track[py, px].astype(np.float32),
            "color_score": color[py, px].astype(np.float32),
            "topology_score": np.ones((len(picked),), np.float32),
        })
        features["sampling_score"] = (features["edge_strength"] + features["trackability_score"] + features["color_score"]).astype(np.float32)
    return {
        "points_xy": picked,
        "features": features,
        "stats": {
            "target_points": int(target),
            "final_sampled_count": int(len(picked)),
            "candidate_count": int(len(points)),
            "bbox_xyxy": bbox.astype(float).tolist(),
            "valid_region_bbox_xyxy": [int(v) for v in valid_bbox],
            "edge_constraint_used": bool(used_edge_constraint),
            "constraint_fallback_level": constraint_fallback_level,
        },
    }
