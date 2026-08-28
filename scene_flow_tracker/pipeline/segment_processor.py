from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch

from ..jobs import DecodedSegment, SegmentResult
from ..query_groups import merge_queries, split_by_layout
from .model_loader import ModelBundle


def detect_bboxes(model: Any, image_rgb: np.ndarray, conf: float) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    result = model.predict(source=image_rgb, conf=conf, verbose=False)[0]
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = [int(round(v)) for v in box.xyxy[0].detach().cpu().numpy().tolist()]
            score = float(box.conf[0].detach().cpu().item())
            cx = 0.5 * (xyxy[0] + xyxy[2])
            dets.append({"bbox_xyxy": xyxy, "confidence": score, "cx": cx})
    if len(dets) < 2:
        raise RuntimeError(f"YOLO detected {len(dets)} boxes, need at least 2")
    top2 = sorted(dets, key=lambda d: d["confidence"], reverse=True)[:2]
    top2 = sorted(top2, key=lambda d: d["cx"])
    top2[0]["slot"] = "left"
    top2[1]["slot"] = "right"
    return top2[0]["bbox_xyxy"], top2[1]["bbox_xyxy"], top2


def frames_to_tensor(frames_bgr: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    return torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)[None].float().to(device)


def normalize_cotracker_output(output: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    confidence = None
    if isinstance(output, dict):
        tracks = output.get("pred_tracks")
        visibility = output.get("pred_visibility")
        confidence = output.get("pred_confidence")
    else:
        tracks, visibility = output[:2]
        if len(output) > 2:
            confidence = output[2]
    tracks_np = tracks.detach().cpu().numpy() if isinstance(tracks, torch.Tensor) else np.asarray(tracks)
    vis_np = visibility.detach().cpu().numpy() if isinstance(visibility, torch.Tensor) else np.asarray(visibility)
    conf_np = confidence.detach().cpu().numpy() if isinstance(confidence, torch.Tensor) else (np.asarray(confidence) if confidence is not None else None)
    if tracks_np.ndim == 4:
        tracks_np = tracks_np[0]
    if vis_np.ndim == 3:
        vis_np = vis_np[0]
    if conf_np is not None and conf_np.ndim == 3:
        conf_np = conf_np[0]
    return tracks_np.astype(np.float32), vis_np.astype(bool), conf_np


def run_cotracker_once(bundle: ModelBundle, frames_bgr: list[np.ndarray], points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(points_xy) == 0:
        t = len(frames_bgr)
        return np.empty((0, t, 2), np.float32), np.empty((0, t), bool), None
    video = frames_to_tensor(frames_bgr, bundle.device)
    all_tracks, all_vis, all_conf = [], [], []
    for start in range(0, len(points_xy), bundle.point_batch_size):
        end = min(start + bundle.point_batch_size, len(points_xy))
        q = np.zeros((1, end - start, 3), dtype=np.float32)
        q[0, :, 0] = 0.0
        q[0, :, 1:] = points_xy[start:end]
        queries = torch.from_numpy(q).to(bundle.device)
        with torch.inference_mode():
            output = bundle.cotracker(video, queries=queries)
        tracks, vis, conf = normalize_cotracker_output(output)
        all_tracks.append(tracks)
        all_vis.append(vis)
        if conf is not None:
            all_conf.append(conf)
    tracks_tn = np.concatenate(all_tracks, axis=1)
    vis_tn = np.concatenate(all_vis, axis=1)
    conf_tn = np.concatenate(all_conf, axis=1) if all_conf else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.transpose(tracks_tn, (1, 0, 2)), np.transpose(vis_tn, (1, 0)), (np.transpose(conf_tn, (1, 0)) if conf_tn is not None else None)


def process_segment(decoded: DecodedSegment, cfg: dict, bundle: ModelBundle) -> SegmentResult:
    job = decoded.job
    timings: dict[str, float] = {"decode_time_sec": decoded.decode_time_sec}
    try:
        total_started = time.perf_counter()
        first_rgb = cv2.cvtColor(decoded.frames_bgr[0], cv2.COLOR_BGR2RGB)

        started = time.perf_counter()
        left_bbox, right_bbox, detections = detect_bboxes(bundle.yolo, first_rgb, float(cfg["models"]["yolo"]["confidence_threshold"]))
        timings["yolo_time_sec"] = time.perf_counter() - started

        started = time.perf_counter()
        # RobotPointSampler computes the full-frame edge bundle once and shares
        # it between LEFT and RIGHT. Environment sampling is routed through the
        # existing EnvironmentPointSampler adapter.
        robot_sample = bundle.robot_sampler.sample(first_rgb, left_bbox, right_bbox)
        env_sample = bundle.env_sampler.sample(first_rgb, left_bbox, right_bbox) if cfg.get("environment_sampling", {}).get("enabled", True) else None
        timings["sampling_time_sec"] = time.perf_counter() - started

        left_q = robot_sample["left_robot"]["points_xy"].astype(np.float32)
        right_q = robot_sample["right_robot"]["points_xy"].astype(np.float32)
        env_q = env_sample["points_xy"].astype(np.float32) if env_sample is not None else np.empty((0, 2), dtype=np.float32)
        all_q, layout = merge_queries({"left": left_q, "right": right_q, "environment": env_q})

        started = time.perf_counter()
        tracks, vis, conf = run_cotracker_once(bundle, decoded.frames_bgr, all_q)
        timings["cotracker_time_sec"] = time.perf_counter() - started

        split_tracks = split_by_layout(tracks, layout)
        split_vis = split_by_layout(vis, layout)
        split_conf = split_by_layout(conf, layout) if conf is not None else {"left": None, "right": None, "environment": None}
        bbox_for = {"left": left_bbox, "right": right_bbox, "environment": [0, 0, first_rgb.shape[1] - 1, first_rgb.shape[0] - 1]}
        query_for = {"left": left_q, "right": right_q, "environment": env_q}
        filter_cfg = cfg.get("trajectory_filter") or cfg.get("cotracker_filter") or {}
        started = time.perf_counter()
        groups = {
            name: bundle.filter_fn(split_tracks[name], split_vis[name], query_for[name], bbox_for[name], filter_cfg, split_conf[name])
            for name in ("left", "right", "environment")
        }
        for name in ("left", "right", "environment"):
            groups[name]["query_xy"] = query_for[name].astype(np.float32)
            if split_conf[name] is not None:
                groups[name]["cotracker_confidence"] = split_conf[name].astype(np.float32)
        timings["trajectory_filter_time_sec"] = time.perf_counter() - started
        timings["segment_total_time_sec"] = time.perf_counter() - total_started
        sampling = {
            "left": robot_sample["left_robot"]["stats"],
            "right": robot_sample["right_robot"]["stats"],
            "environment": env_sample["stats"] if env_sample is not None else {"final_sampled_count": 0},
            "query_group_id": layout.group_id,
            "query_local_point_id": layout.local_point_id,
            "left_bbox": left_bbox,
            "right_bbox": right_bbox,
            "image_width": int(first_rgb.shape[1]),
            "image_height": int(first_rgb.shape[0]),
        }
        return SegmentResult(job=job, status="DONE", detections=detections, sampling=sampling, groups=groups, timings=timings)
    except Exception as exc:
        return SegmentResult(job=job, status="FAILED", error_code=type(exc).__name__, error_message=str(exc), timings=timings)
