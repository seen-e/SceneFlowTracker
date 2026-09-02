from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeJob:
    dataset: str
    episode_id: str
    episode_index: int
    task_index: int | None
    task: str | None
    instruction: str | None
    view_key: str
    physical_video_path: str
    source_start_frame: int
    source_end_frame: int
    manifest_fps: float
    effective_fps: float
    frame_count: int
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentJob:
    dataset: str
    episode_id: str
    episode_index: int
    view_key: str
    physical_video_path: str
    segment_id: int
    episode_start_frame: int
    episode_end_frame: int
    source_start_frame: int
    source_end_frame: int
    frame_count: int
    manifest_fps: float
    effective_fps: float
    content_bbox_xyxy: tuple[int, int, int, int] | None = None


@dataclass
class DecodedSegment:
    job: SegmentJob
    frames_bgr: list[np.ndarray]
    decode_time_sec: float


@dataclass
class SegmentResult:
    job: SegmentJob
    status: str
    error_code: str | None = None
    error_message: str | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode: EpisodeJob
    segments: list[SegmentResult]
