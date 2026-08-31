from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ModelBundle:
    yolo: Any
    cotracker: Any
    device: torch.device
    point_batch_size: int
    robot_sampler: Any
    env_sampler: Any
    filter_fn: Any


def _canonical_device_name(name: str) -> str:
    return str(torch.device(name))


def _assigned_worker_count_for_device(cfg: dict, device_name: str) -> int:
    devices = list(cfg.get("workers", {}).get("model_devices") or ["cpu"])
    model_workers = int(cfg.get("workers", {}).get("model_workers", 1))
    target = _canonical_device_name(device_name)
    count = 0
    for worker_id in range(model_workers):
        assigned = _canonical_device_name(str(devices[worker_id % len(devices)]))
        if assigned == target:
            count += 1
    return max(1, count)


def load_model_bundle(cfg: dict, device_name: str) -> ModelBundle:
    legacy_root = Path(cfg["legacy_modules"]["module_root"])
    cot_root = Path(cfg["models"]["cotracker"]["source_root"])
    for root in (legacy_root, cot_root):
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from ultralytics import YOLO
    from cotracker.predictor import CoTrackerPredictor
    from robot_sampling import EnvironmentPointSampler, RobotPointSampler
    from tracking import filter_robot_tracks

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        workers_cfg = cfg.get("workers", {})
        per_device_limit_gb = workers_cfg.get("gpu_memory_limit_per_device_gb")
        per_worker_limit_gb = workers_cfg.get("gpu_memory_limit_gb")
        if per_device_limit_gb is not None:
            workers_on_device = _assigned_worker_count_for_device(cfg, str(device))
            limit_gb = float(per_device_limit_gb) / workers_on_device
        else:
            limit_gb = per_worker_limit_gb
        if limit_gb is not None:
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            limit_bytes = float(limit_gb) * 1024**3
            fraction = max(0.01, min(1.0, limit_bytes / float(total_bytes)))
            torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    yolo = YOLO(str(cfg["models"]["yolo"]["model_path"]))
    cotracker = CoTrackerPredictor(
        checkpoint=str(cfg["models"]["cotracker"]["model_path"]),
        offline=True,
        window_len=int(cfg["models"]["cotracker"].get("window_len", 60)),
    ).to(device).eval()
    robot_sampler = RobotPointSampler(cfg.get("robot_sampling", {}))
    env_sampler = EnvironmentPointSampler(cfg.get("environment_sampling", {}), cfg.get("robot_sampling", {}))
    return ModelBundle(
        yolo=yolo,
        cotracker=cotracker,
        device=device,
        point_batch_size=int(cfg["models"]["cotracker"].get("point_batch_size", 200)),
        robot_sampler=robot_sampler,
        env_sampler=env_sampler,
        filter_fn=filter_robot_tracks,
    )
