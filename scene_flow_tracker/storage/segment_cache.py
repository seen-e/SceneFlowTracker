from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..jobs import EpisodeJob, SegmentJob, SegmentResult
from .writers import episode_output_dir, json_safe, safe_view_name

CACHE_SCHEMA_VERSION = "segment-cache-v1"


def config_fingerprint(cfg: dict[str, Any]) -> str:
    import hashlib

    relevant = {
        "video": cfg.get("video", {}),
        "models": cfg.get("models", {}),
        "sampling": cfg.get("sampling", {}),
        "robot_sampling": cfg.get("robot_sampling", {}),
        "environment_sampling": cfg.get("environment_sampling", {}),
        "trajectory_filter": cfg.get("trajectory_filter", {}),
        "output": {
            key: value
            for key, value in (cfg.get("output", {}) or {}).items()
            if key not in {"output_root", "debug_visualization"}
        },
    }
    blob = json.dumps(json_safe(relevant), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def segment_cache_dir(output_root: Path, episode: EpisodeJob, cfg: dict[str, Any]) -> Path:
    dirname = str((cfg.get("cache", {}) or {}).get("dirname", ".segment_cache"))
    return episode_output_dir(output_root, episode) / dirname / safe_view_name(episode.view_key)


def segment_cache_path(output_root: Path, episode: EpisodeJob, job: SegmentJob, cfg: dict[str, Any]) -> Path:
    return segment_cache_dir(output_root, episode, cfg) / f"segment_{job.segment_id:06d}.npz"


def _obj(value: Any) -> np.ndarray:
    return np.array(json.dumps(json_safe(value), ensure_ascii=False), dtype=np.str_)


def _load_obj(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data:
        return default
    text = str(data[key].item())
    return json.loads(text) if text else default


def write_segment_cache(path: Path, result: SegmentResult, fingerprint: str, atomic: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cache_schema_version": np.array(CACHE_SCHEMA_VERSION, dtype=np.str_),
        "fingerprint": np.array(fingerprint, dtype=np.str_),
        "created_at": np.array(time.strftime("%Y-%m-%dT%H:%M:%S%z"), dtype=np.str_),
        "job": _obj(result.job.__dict__),
        "status": np.array(result.status, dtype=np.str_),
        "error_code": np.array("" if result.error_code is None else result.error_code, dtype=np.str_),
        "error_message": np.array("" if result.error_message is None else result.error_message, dtype=np.str_),
        "detections_json": _obj(result.detections),
        "sampling_json": _obj(result.sampling),
        "timings_json": _obj(result.timings),
        "groups_json": _obj(
            {
                name: {
                    key: value
                    for key, value in group.items()
                    if not isinstance(value, np.ndarray)
                }
                for name, group in result.groups.items()
            }
        ),
    }
    for group_name, group in result.groups.items():
        prefix = f"group_{group_name}_"
        for key, value in group.items():
            if isinstance(value, np.ndarray):
                payload[prefix + key] = value
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    if atomic:
        os.replace(tmp, path)
    else:
        shutil.move(tmp, path)


def read_segment_cache(path: Path) -> tuple[SegmentResult, str]:
    with np.load(path, allow_pickle=False) as data:
        version = str(data["cache_schema_version"].item())
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported segment cache schema {version}: {path}")
        fingerprint = str(data["fingerprint"].item())
        job_payload = _load_obj(data, "job", {})
        job = SegmentJob(**job_payload)
        groups_meta = _load_obj(data, "groups_json", {})
        groups: dict[str, dict[str, Any]] = {}
        for group_name, meta in groups_meta.items():
            group = dict(meta)
            prefix = f"group_{group_name}_"
            for key in data.files:
                if key.startswith(prefix):
                    group[key[len(prefix) :]] = data[key].copy()
            groups[group_name] = group
        return (
            SegmentResult(
                job=job,
                status=str(data["status"].item()),
                error_code=str(data["error_code"].item()) or None,
                error_message=str(data["error_message"].item()) or None,
                detections=_load_obj(data, "detections_json", []),
                sampling=_load_obj(data, "sampling_json", {}),
                groups=groups,
                timings=_load_obj(data, "timings_json", {}),
            ),
            fingerprint,
        )


def validate_segment_cache(path: Path, job: SegmentJob, fingerprint: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        result, cached_fp = read_segment_cache(path)
    except Exception:
        return False
    return (
        cached_fp == fingerprint
        and result.job.segment_id == job.segment_id
        and result.job.episode_id == job.episode_id
        and result.job.source_start_frame == job.source_start_frame
        and result.job.source_end_frame == job.source_end_frame
    )


def delete_segment_cache_dir(output_root: Path, episode: EpisodeJob, cfg: dict[str, Any]) -> None:
    root = segment_cache_dir(output_root, episode, cfg)
    if root.exists():
        shutil.rmtree(root)
