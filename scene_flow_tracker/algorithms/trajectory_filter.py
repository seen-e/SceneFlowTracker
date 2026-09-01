from __future__ import annotations

import numpy as np

from .trajectory_features import trajectory_features
from .trajectory_smoothing import smooth_tracks


def _bbox_for_group(group_name: str, detections) -> np.ndarray | None:
    if detections is None:
        return None
    if group_name == "left" and detections.left_bbox_valid:
        return detections.left_bbox_xyxy
    if group_name == "right" and detections.right_bbox_valid:
        return detections.right_bbox_xyxy
    return None


def filter_tracks_for_group(tracks_xy: np.ndarray, visibility: np.ndarray, query_xy: np.ndarray, bbox_xyxy: np.ndarray | None, cfg: dict, confidence: np.ndarray | None = None) -> dict:
    n = len(query_xy)
    smooth = smooth_tracks(tracks_xy, cfg.get("smoothing", {}))
    features = trajectory_features(smooth, visibility, bbox_xyxy)
    track_state = np.asarray(["valid"] * n, dtype="<U16")
    if bool(cfg.get("tracking_validity", {}).get("enabled", False)):
        min_vis = float(cfg.get("tracking_validity", {}).get("min_visibility_ratio", 0.6))
        partial_vis = float(cfg.get("tracking_validity", {}).get("partial_visibility_ratio", 0.3))
        vr = features["visibility_ratio"]
        track_state[vr < partial_vis] = "failed"
        track_state[(vr >= partial_vis) & (vr < min_vis)] = "partial"
    mc = cfg.get("motion_classification", {})
    static_cfg = mc.get("static", {})
    jitter_cfg = mc.get("jitter_v2", {})
    moving_cfg = mc.get("moving", {})
    motion = np.asarray(["uncertain"] * n, dtype="<U16")
    static = (
        (features["net_displacement"] <= float(static_cfg.get("max_net_displacement_norm", 0.01)))
        & (features["path_length"] <= float(static_cfg.get("max_path_length_norm", 0.02)))
        & (features["jitter_rms"] <= float(static_cfg.get("max_jitter_rms_norm", 0.005)))
    )
    moving = (
        (features["net_displacement"] >= float(moving_cfg.get("min_net_displacement_norm", 0.01)))
        | (features["path_length"] >= float(mc.get("structured_motion", {}).get("min_path_length_norm", 0.015)))
    )
    evidence = np.zeros((n,), np.int16)
    residual_cfg = jitter_cfg.get("residual_ratio", {})
    if residual_cfg.get("enabled", True):
        evidence += features["jitter_residual_ratio"] >= float(residual_cfg.get("min_jitter_residual_ratio", 0.35))
    turning_cfg = jitter_cfg.get("turning", {})
    if turning_cfg.get("enabled", True):
        evidence += (
            (features["turn_consistency"] <= float(turning_cfg.get("max_turn_consistency", 0.35)))
            & (features["turn_angle_mad"] >= float(turning_cfg.get("min_turn_angle_mad", 0.75)))
        )
    jerk_cfg = jitter_cfg.get("jerk", {})
    if jerk_cfg.get("enabled", True):
        evidence += features["normalized_jerk"] >= float(jerk_cfg.get("min_normalized_jerk", 1.2))
    jitter = evidence >= int(jitter_cfg.get("min_evidence_count", 2))
    motion[static] = "static"
    motion[moving & ~jitter] = "moving"
    motion[jitter & ~static] = "jitter"
    policy = cfg.get("output_policy", {})
    usable = np.zeros((n,), dtype=bool)
    usable |= (motion == "static") & bool(policy.get("keep_static", True))
    usable |= (motion == "moving") & bool(policy.get("keep_moving", True))
    usable |= (motion == "jitter") & bool(policy.get("keep_jitter", False))
    usable |= (motion == "uncertain") & bool(policy.get("keep_uncertain", True))
    usable &= track_state == "valid"
    if bool(policy.get("keep_partial_tracks", True)):
        usable |= track_state == "partial"
    out = {
        "query_xy": query_xy.astype(np.float32),
        "tracks_xy_raw": tracks_xy.astype(np.float32),
        "tracks_xy_smooth": smooth.astype(np.float32),
        "visibility": visibility.astype(bool),
        "track_state": track_state,
        "motion_state": motion,
        "usable_for_robot_scene_flow": usable,
    }
    out.update(features)
    if confidence is not None:
        out["cotracker_confidence"] = confidence.astype(np.float32)
    return out


def filter_tracks(track_result, cfg: dict) -> dict[str, dict]:
    q = track_result.query_xy
    groups = {}
    for name, gid in (("left", 0), ("right", 1), ("environment", 2)):
        idx = np.nonzero(track_result.query_group == gid)[0]
        bbox = _bbox_for_group(name, track_result.detections)
        conf = track_result.confidence[idx] if track_result.confidence is not None else None
        groups[name] = filter_tracks_for_group(
            track_result.tracks_xy[idx],
            track_result.visibility[idx],
            q[idx],
            bbox,
            cfg,
            conf,
        )
    return groups
