from __future__ import annotations

import numpy as np


def trajectory_features(tracks_xy: np.ndarray, visibility: np.ndarray, bbox_xyxy: np.ndarray | None) -> dict[str, np.ndarray]:
    tracks = np.asarray(tracks_xy, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    n, t = tracks.shape[:2]
    if n == 0:
        empty = np.empty((0,), np.float32)
        return {k: empty for k in ("visibility_ratio", "net_displacement", "path_length", "path_efficiency", "jitter_rms", "jitter_residual_ratio", "turn_consistency", "turn_angle_mad", "normalized_jerk", "direction_reversal_ratio")}
    diffs = np.diff(tracks, axis=1) if t > 1 else np.zeros((n, 0, 2), np.float32)
    step = np.linalg.norm(diffs, axis=2) if t > 1 else np.zeros((n, 0), np.float32)
    net = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=1) if t else np.zeros((n,), np.float32)
    path = step.sum(axis=1)
    efficiency = net / np.maximum(path, 1e-6)
    linear = tracks[:, :1] + (tracks[:, -1:] - tracks[:, :1]) * (np.linspace(0, 1, max(t, 1), dtype=np.float32)[None, :, None])
    residual = np.linalg.norm(tracks - linear, axis=2)
    jitter_rms = np.sqrt(np.mean(residual**2, axis=1)) if t else np.zeros((n,), np.float32)
    residual_ratio = jitter_rms / np.maximum(path, 1e-6)
    signs = np.sign(diffs)
    flips = (signs[:, 1:] * signs[:, :-1] < 0).any(axis=2) if t > 2 else np.zeros((n, 0), bool)
    reversal = flips.mean(axis=1) if flips.shape[1] else np.zeros((n,), np.float32)
    angles = np.zeros((n, max(0, t - 2)), np.float32)
    if t > 2:
        a = diffs[:, :-1]
        b = diffs[:, 1:]
        denom = np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-6)
        cos = np.clip((a * b).sum(axis=2) / denom, -1.0, 1.0)
        angles = np.arccos(cos).astype(np.float32)
    turn_consistency = 1.0 - np.minimum(1.0, angles.mean(axis=1) / np.pi) if angles.shape[1] else np.ones((n,), np.float32)
    turn_mad = np.median(np.abs(angles - np.median(angles, axis=1, keepdims=True)), axis=1) if angles.shape[1] else np.zeros((n,), np.float32)
    jerk = np.diff(diffs, n=2, axis=1) if t > 3 else np.zeros((n, 0, 2), np.float32)
    jerk_norm = np.linalg.norm(jerk, axis=2).mean(axis=1) if jerk.shape[1] else np.zeros((n,), np.float32)
    if bbox_xyxy is not None:
        b = np.asarray(bbox_xyxy, dtype=np.float32)
        norm = float(np.hypot(max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])))
    else:
        norm = 1.0
    return {
        "visibility_ratio": vis.mean(axis=1).astype(np.float32) if t else np.zeros((n,), np.float32),
        "net_displacement": (net / norm).astype(np.float32),
        "path_length": (path / norm).astype(np.float32),
        "path_efficiency": efficiency.astype(np.float32),
        "jitter_rms": (jitter_rms / norm).astype(np.float32),
        "jitter_residual_ratio": residual_ratio.astype(np.float32),
        "turn_consistency": turn_consistency.astype(np.float32),
        "turn_angle_mad": turn_mad.astype(np.float32),
        "normalized_jerk": (jerk_norm / norm).astype(np.float32),
        "direction_reversal_ratio": reversal.astype(np.float32),
    }
