from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..jobs import EpisodeJob, SegmentJob


@dataclass(frozen=True)
class EndOfStream:
    source: str


@dataclass(frozen=True)
class WorkerFailure:
    source: str
    error_code: str
    error_message: str
    fatal: bool = False
    job: SegmentJob | None = None


@dataclass(frozen=True)
class SharedArrayRef:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    owner: str
    debug_id: str


@dataclass
class FirstFrameItem:
    job: SegmentJob
    frame_rgb: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class YoloDetectionResult:
    job: SegmentJob
    first_frame_rgb: np.ndarray
    left_bbox_xyxy: np.ndarray | None
    left_bbox_valid: bool
    left_confidence: float
    right_bbox_xyxy: np.ndarray | None
    right_bbox_valid: bool
    right_confidence: float
    raw_detections: list[dict[str, Any]]
    assignment_method: str
    image_height: int
    image_width: int
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class SamplingResult:
    job: SegmentJob
    detections: YoloDetectionResult
    query_xy: np.ndarray
    query_group: np.ndarray
    left_count: int
    right_count: int
    env_count: int
    sampling_features: dict[str, Any]
    sampling_stats: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class DecodedTrackItem:
    job: SegmentJob
    frame_ref: SharedArrayRef
    query_xy: np.ndarray
    query_group: np.ndarray
    detections: YoloDetectionResult
    sampling_features: dict[str, Any]
    sampling_stats: dict[str, Any]
    image_height: int
    image_width: int
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class TrackResult:
    job: SegmentJob
    tracks_xy: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray | None
    query_xy: np.ndarray
    query_group: np.ndarray
    detections: YoloDetectionResult
    sampling_features: dict[str, Any]
    sampling_stats: dict[str, Any]
    image_height: int
    image_width: int
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class FilteredSegmentResult:
    job: SegmentJob
    status: str
    error_code: str | None = None
    error_message: str | None = None
    detections: YoloDetectionResult | None = None
    query_xy: np.ndarray | None = None
    query_group: np.ndarray | None = None
    left_count: int = 0
    right_count: int = 0
    env_count: int = 0
    sampling_features: dict[str, Any] = field(default_factory=dict)
    sampling_stats: dict[str, Any] = field(default_factory=dict)
    tracks_xy_raw: np.ndarray | None = None
    tracks_xy_smooth: np.ndarray | None = None
    visibility: np.ndarray | None = None
    confidence: np.ndarray | None = None
    track_state: np.ndarray | None = None
    motion_state: np.ndarray | None = None
    usable: np.ndarray | None = None
    filter_features: dict[str, np.ndarray] = field(default_factory=dict)
    image_height: int = -1
    image_width: int = -1
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class YoloBatch:
    items: list[FirstFrameItem]
    batch_size: int
    fill_ratio: float


@dataclass
class YoloBatchResult:
    results: list[YoloDetectionResult]
    batch_size: int
    fill_ratio: float
    forward_time_sec: float
    gpu_peak_memory_mb: float | None = None


@dataclass
class CoTrackerBatch:
    items: list[DecodedTrackItem]
    batch_key: tuple[int, int, int, int]
    batch_size: int
    fill_ratio: float
    is_tail: bool = False


@dataclass
class CoTrackerBatchResult:
    results: list[TrackResult]
    batch_size: int
    fill_ratio: float
    forward_time_sec: float
    gpu_peak_memory_mb: float | None = None


__all__ = [
    "EpisodeJob",
    "SegmentJob",
    "EndOfStream",
    "WorkerFailure",
    "SharedArrayRef",
    "FirstFrameItem",
    "YoloDetectionResult",
    "SamplingResult",
    "DecodedTrackItem",
    "TrackResult",
    "FilteredSegmentResult",
    "YoloBatch",
    "YoloBatchResult",
    "CoTrackerBatch",
    "CoTrackerBatchResult",
]
