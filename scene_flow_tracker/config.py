from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .model_parallel import validate_model_parallel_config


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "manifest_path": "",
        "view_key": "observation.images.top",
        "frame_range_semantics": "half_open",
        "strict_validation": True,
    },
    "video": {
        "fps_mode": "manifest",
        "fixed_fps": 30.0,
        "segment_frames": 15,
        "tail_policy": "keep",
    },
    "models": {
        "yolo": {
            "model_path": "",
            "devices": [0],
            "worker_count": 1,
            "batch_size": 32,
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.7,
        },
        "cotracker": {
            "source_root": "/mnt/data/chachaxu/instance_tracking_environment/co-tracker",
            "model_path": "",
            "window_len": 60,
            "devices": [1],
            "worker_count": 1,
            "segment_batch_size": 4,
            "point_chunk_size": 1024,
        },
    },
    "workers": {
        "resume_scan_workers": 16,
        "first_frame_decode_workers": 8,
        "sampling_workers": 8,
        "segment_decode_workers": 4,
        "filter_workers": 8,
    },
    "pipeline": {
        "max_inflight_segments": 64,
    },
    "queues": {
        "segment_job_queue_size": 128,
        "first_frame_queue_size": 64,
        "yolo_batch_queue_size": 4,
        "yolo_result_queue_size": 64,
        "sampling_result_queue_size": 32,
        "decoded_track_queue_size": 8,
        "cotracker_batch_queue_size": 2,
        "track_result_queue_size": 16,
        "filtered_result_queue_size": 32,
    },
    "processing": {"depth_enabled": False},
    "depth_sampling": {
        "enabled": False,
        "root": "",
        "fallback_to_rgb_when_missing": True,
        "gradient_percentile": 85.0,
        "min_gradient": 0.02,
        "dilate_px": 1,
    },
    "sampling": {
        "seed": 0,
        "query_allocation": {
            "total_query_points": 300,
            "points_per_detected_arm": 100,
        },
    },
    "cache": {
        "enabled": True,
        "dirname": ".segment_cache",
        "delete_after_successful_merge": True,
        "retry_cached_failed_segments": False,
    },
    "robot_sampling": {},
    "environment_sampling": {"enabled": True, "target_points": 300},
    "trajectory_filter": {},
    "output": {
        "output_root": "./outputs",
        "schema_version": "1.2",
        "save_npz": True,
        "save_summary_json": True,
        "compression": "compressed",
        "save_raw_tracks": True,
        "save_smooth_tracks": True,
        "save_features": True,
        "save_sampling_features": True,
        "save_filter_features": True,
        "save_cotracker_confidence": True,
        "debug_visualization": False,
    },
    "batch": {
        "resume": True,
        "continue_on_segment_error": True,
        "group_by_physical_video": True,
        "atomic_write": True,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    includes = payload.pop("algorithm_configs", {}) or {}
    include_payload: dict[str, Any] = {}
    for key, include_path in includes.items():
        inc = Path(include_path)
        if not inc.is_absolute():
            inc = path.parent / inc
        data = yaml.safe_load(inc.read_text(encoding="utf-8")) or {}
        if key == "sampling":
            include_payload = deep_update(include_payload, {
                "sampling": data.get("sampling", {}),
                "robot_sampling": data.get("robot_sampling", {}),
                "environment_sampling": data.get("environment_sampling", {}),
            })
        elif key in {"trajectory_filter", "filter"}:
            include_payload = deep_update(include_payload, {
                "trajectory_filter": data.get("trajectory_filter", data),
            })
        else:
            include_payload = deep_update(include_payload, {key: data})
    cfg = deep_update(DEFAULT_CONFIG, include_payload)
    cfg = deep_update(cfg, payload)
    cfg["algorithm_configs"] = {k: str(v) for k, v in includes.items()}
    cfg["_config_path"] = str(path)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    manifest = Path(cfg["input"]["manifest_path"])
    if not manifest.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest}")
    if cfg["input"].get("frame_range_semantics") != "half_open":
        raise ValueError("Only half_open frame ranges are supported")
    if int(cfg["video"]["segment_frames"]) <= 0:
        raise ValueError("video.segment_frames must be > 0")
    if cfg["video"]["fps_mode"] not in {"manifest", "fixed"}:
        raise ValueError("video.fps_mode must be manifest or fixed")
    if cfg["video"]["tail_policy"] not in {"keep", "drop"}:
        raise ValueError("video.tail_policy must be keep or drop")
    for key in ("resume_scan_workers", "first_frame_decode_workers", "sampling_workers", "segment_decode_workers", "filter_workers"):
        if int(cfg["workers"][key]) < 1:
            raise ValueError(f"workers.{key} must be >= 1")
    if int((cfg.get("pipeline", {}) or {}).get("max_inflight_segments", 64)) < 1:
        raise ValueError("pipeline.max_inflight_segments must be >= 1")
    sampling = (cfg.get("sampling", {}) or {}).get("query_allocation", {})
    total = int(sampling.get("total_query_points", 0))
    per_arm = int(sampling.get("points_per_detected_arm", 0))
    if total <= 0:
        raise ValueError("sampling.query_allocation.total_query_points must be > 0")
    if per_arm < 0:
        raise ValueError("sampling.query_allocation.points_per_detected_arm must be >= 0")
    if per_arm * 2 > total:
        raise ValueError("sampling.query_allocation.points_per_detected_arm * 2 must be <= total_query_points")
    validate_model_parallel_config("yolo", cfg["models"]["yolo"])
    validate_model_parallel_config("cotracker", cfg["models"]["cotracker"])
    for key, value in cfg["queues"].items():
        if int(value) <= 0:
            raise ValueError(f"queues.{key} must be > 0")
    yolo = Path(cfg["models"]["yolo"]["model_path"])
    cot = Path(cfg["models"]["cotracker"]["model_path"])
    if not yolo.exists():
        raise FileNotFoundError(f"YOLO model_path does not exist: {yolo}")
    if not cot.exists():
        raise FileNotFoundError(f"CoTracker model_path does not exist: {cot}")
    Path(cfg["output"]["output_root"]).mkdir(parents=True, exist_ok=True)
    if cfg["processing"].get("depth_enabled", False):
        raise ValueError("processing.depth_enabled must remain false in this framework version")
    for deprecated in ("model_workers", "model_devices", "allow_device_sharing", "decode_workers"):
        if deprecated in cfg["workers"]:
            logging.warning("workers.%s is deprecated and ignored by the refactored pipeline", deprecated)
    if "legacy_modules" in cfg:
        logging.warning("legacy_modules is deprecated and ignored by the refactored pipeline")
