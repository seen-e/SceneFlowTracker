from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


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
        "yolo": {"model_path": "", "confidence_threshold": 0.15},
        "cotracker": {
            "source_root": "/mnt/data/chachaxu/instance_tracking_environment/co-tracker",
            "model_path": "",
            "window_len": 60,
            "point_batch_size": 200,
        },
    },
    "workers": {
        "decode_workers": 4,
        "model_workers": 1,
        "model_devices": ["cuda:0"],
        "allow_device_sharing": False,
        "gpu_memory_limit_gb": None,
        "gpu_memory_limit_per_device_gb": None,
    },
    "queues": {
        "segment_job_queue_size": 64,
        "decoded_segment_queue_size": 16,
        "result_queue_size": 64,
    },
    "processing": {"depth_enabled": False},
    "legacy_modules": {
        "module_root": "/mnt/workspace/instance_exp/scen_flow_main_view",
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
                "robot_sampling": data.get("robot_sampling", {}),
                "environment_sampling": data.get("environment_sampling", {}),
            })
        elif key in {"trajectory_filter", "filter"}:
            include_payload = deep_update(include_payload, {
                "trajectory_filter": data.get("trajectory_filter", data),
            })
        else:
            include_payload = deep_update(include_payload, {key: data})
    cfg = deep_update(DEFAULT_CONFIG, payload)
    cfg = deep_update(cfg, include_payload)
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
    if int(cfg["workers"]["decode_workers"]) < 1:
        raise ValueError("workers.decode_workers must be >= 1")
    if int(cfg["workers"]["model_workers"]) < 1:
        raise ValueError("workers.model_workers must be >= 1")
    devices = list(cfg["workers"].get("model_devices") or [])
    model_workers = int(cfg["workers"]["model_workers"])
    if len(devices) < model_workers and not cfg["workers"].get("allow_device_sharing", False):
        raise ValueError("model_workers > len(model_devices); enable allow_device_sharing to share devices")
    gpu_memory_limit_gb = cfg["workers"].get("gpu_memory_limit_gb")
    if gpu_memory_limit_gb is not None and float(gpu_memory_limit_gb) <= 0:
        raise ValueError("workers.gpu_memory_limit_gb must be positive or null")
    gpu_memory_limit_per_device_gb = cfg["workers"].get("gpu_memory_limit_per_device_gb")
    if gpu_memory_limit_per_device_gb is not None and float(gpu_memory_limit_per_device_gb) <= 0:
        raise ValueError("workers.gpu_memory_limit_per_device_gb must be positive or null")
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
    if len(devices) > model_workers:
        logging.warning("More model_devices than model_workers; extra devices will be unused")
