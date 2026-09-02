from __future__ import annotations

from typing import Any


def normalize_device(value: Any) -> str:
    """Normalize compact GPU ids from config into framework device strings."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"GPU device id must be >= 0, got {value}")
        return f"cuda:{value}"
    text = str(value).strip()
    if not text:
        raise ValueError("device must not be empty")
    if text.isdigit():
        return f"cuda:{int(text)}"
    return text


def configured_devices(model_cfg: dict[str, Any]) -> list[str]:
    """Return the configured model worker device list."""
    raw = model_cfg.get("devices")
    if raw is None:
        raise ValueError("models.<name>.devices must be configured")
    if isinstance(raw, str):
        devices = [normalize_device(item) for item in raw.split(",") if item.strip()]
    else:
        devices = [normalize_device(item) for item in raw]
    if not devices:
        raise ValueError("models.<name>.devices must contain at least one device")
    return devices


def worker_count(model_cfg: dict[str, Any]) -> int:
    return int(model_cfg.get("worker_count", 1))


def expanded_worker_devices(model_cfg: dict[str, Any]) -> list[str]:
    devices = configured_devices(model_cfg)
    count = worker_count(model_cfg)
    return [devices[idx % len(devices)] for idx in range(count)]


def validate_model_parallel_config(name: str, model_cfg: dict[str, Any]) -> None:
    try:
        devices = configured_devices(model_cfg)
    except ValueError as exc:
        raise ValueError(str(exc).replace("models.<name>", f"models.{name}")) from exc
    count = worker_count(model_cfg)
    if count < 1:
        raise ValueError(f"models.{name}.worker_count must be >= 1")
    if count % len(devices) != 0:
        raise ValueError(
            f"models.{name}.worker_count ({count}) must be divisible by "
            f"len(models.{name}.devices) ({len(devices)})"
        )
