from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .jobs import EpisodeJob


def _segment_for(record: dict[str, Any], view_key: str) -> dict[str, Any]:
    try:
        return record["video_segments"][view_key]
    except KeyError as exc:
        raise KeyError(f"video_segments missing view_key={view_key} for episode_id={record.get('episode_id')}") from exc


def load_episode_jobs(cfg: dict[str, Any]) -> tuple[list[EpisodeJob], list[dict[str, Any]]]:
    manifest_path = Path(cfg["input"]["manifest_path"])
    view_key = cfg["input"]["view_key"]
    strict = bool(cfg["input"].get("strict_validation", True))
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs: list[EpisodeJob] = []
    invalid: list[dict[str, Any]] = []
    for record in records:
        try:
            seg = _segment_for(record, view_key)
            source_start = int(seg["start_frame"])
            source_end = int(seg["end_frame"])
            if source_end <= source_start:
                raise ValueError("end_frame must be > start_frame")
            frame_count = int(record.get("frame_count", source_end - source_start))
            if source_end - source_start != frame_count:
                raise ValueError(f"FRAME_COUNT_MISMATCH: video_segments has {source_end-source_start}, record has {frame_count}")
            manifest_fps = float(seg.get("fps", record.get("fps", 0.0)))
            if cfg["video"]["fps_mode"] == "fixed":
                effective_fps = float(cfg["video"]["fixed_fps"])
                if abs(effective_fps - manifest_fps) > 1e-3:
                    logging.warning("fixed_fps %.3f differs from manifest_fps %.3f for %s", effective_fps, manifest_fps, record.get("episode_id"))
            else:
                effective_fps = manifest_fps
            jobs.append(
                EpisodeJob(
                    dataset=str(record.get("dataset", "unknown_dataset")),
                    episode_id=str(record["episode_id"]),
                    episode_index=int(record["episode_index"]),
                    task_index=int(record["task_index"]) if record.get("task_index") is not None else None,
                    task=record.get("task"),
                    instruction=record.get("instruction"),
                    view_key=view_key,
                    physical_video_path=str(seg["video_path"]),
                    source_start_frame=source_start,
                    source_end_frame=source_end,
                    manifest_fps=manifest_fps,
                    effective_fps=effective_fps,
                    frame_count=frame_count,
                    raw_record=record,
                )
            )
        except Exception as exc:
            item = {"episode_id": record.get("episode_id"), "episode_index": record.get("episode_index"), "error": str(exc)}
            invalid.append(item)
            if strict:
                raise
    if cfg["batch"].get("group_by_physical_video", True):
        jobs.sort(key=lambda j: (j.physical_video_path, j.source_start_frame, j.episode_index))
    else:
        jobs.sort(key=lambda j: j.episode_index)
    return jobs, invalid
