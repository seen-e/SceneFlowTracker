from __future__ import annotations

from typing import Any


def configured_devices(model_cfg: dict[str, Any], default_device: str) -> list[str]:
    """Return the configured device list, preserving the legacy device field."""
    raw = model_cfg.get("devices")
    if raw is None:
        raw = [model_cfg.get("device", default_device)]
    if isinstance(raw, str):
        devices = [raw]
    else:
        devices = [str(item) for item in raw]
    devices = [device for device in devices if device]
    return devices or [default_device]


def worker_count(model_cfg: dict[str, Any]) -> int:
    return int(model_cfg.get("worker_count", 1))


def expanded_worker_devices(model_cfg: dict[str, Any], default_device: str) -> list[str]:
    devices = configured_devices(model_cfg, default_device)
    count = worker_count(model_cfg)
    return [devices[idx % len(devices)] for idx in range(count)]


def validate_model_parallel_config(name: str, model_cfg: dict[str, Any], default_device: str) -> None:
    devices = configured_devices(model_cfg, default_device)
    count = worker_count(model_cfg)
    if count < 1:
        raise ValueError(f"models.{name}.worker_count must be >= 1")
    if count % len(devices) != 0:
        raise ValueError(
            f"models.{name}.worker_count ({count}) must be divisible by "
            f"len(models.{name}.devices) ({len(devices)})"
        )
