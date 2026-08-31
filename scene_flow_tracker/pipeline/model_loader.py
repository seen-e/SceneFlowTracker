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
        limit_gb = cfg.get("workers", {}).get("gpu_memory_limit_gb")
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
