from __future__ import annotations

from typing import Any

import numpy as np

from ..data.types import FilteredSegmentResult, TrackResult, YoloDetectionResult
from ..jobs import SegmentJob, SegmentResult
from ..storage.schema import GROUP_TO_RESULT_KEY


def failed_segment_result(job: SegmentJob, error_code: str, error_message: str, timings: dict[str, float] | None = None) -> SegmentResult:
    return SegmentResult(
        job=job,
        status="FAILED",
        error_code=error_code,
        error_message=error_message,
        timings=timings or {},
    )


def filtered_failure(job: SegmentJob, error_code: str, error_message: str, timings: dict[str, float] | None = None) -> FilteredSegmentResult:
    return FilteredSegmentResult(
        job=job,
        status="FAILED",
        error_code=error_code,
        error_message=error_message,
        timings=timings or {},
    )


def detections_to_list(det: YoloDetectionResult | None) -> list[dict[str, Any]]:
    if det is None:
        return []
    out: list[dict[str, Any]] = []
    if det.left_bbox_valid and det.left_bbox_xyxy is not None:
        out.append(
            {
                "slot": "left",
                "bbox_xyxy": np.asarray(det.left_bbox_xyxy, dtype=np.float32).tolist(),
                "confidence": float(det.left_confidence),
                "class_id": 0,
            }
        )
    if det.right_bbox_valid and det.right_bbox_xyxy is not None:
        out.append(
            {
                "slot": "right",
                "bbox_xyxy": np.asarray(det.right_bbox_xyxy, dtype=np.float32).tolist(),
                "confidence": float(det.right_confidence),
                "class_id": 1,
            }
        )
    return out


def sampling_summary(filtered: FilteredSegmentResult) -> dict[str, Any]:
    det = filtered.detections
    return {
        "image_width": int(filtered.image_width),
        "image_height": int(filtered.image_height),
        "query_group_id": None if filtered.query_group is None else filtered.query_group.astype(np.int16).tolist(),
        "left_bbox": None if det is None or not det.left_bbox_valid or det.left_bbox_xyxy is None else np.asarray(det.left_bbox_xyxy, dtype=np.float32).tolist(),
        "right_bbox": None if det is None or not det.right_bbox_valid or det.right_bbox_xyxy is None else np.asarray(det.right_bbox_xyxy, dtype=np.float32).tolist(),
        "yolo_assignment_method": None if det is None else det.assignment_method,
        "yolo_raw_detections": [] if det is None else det.raw_detections,
        "left_count": int(filtered.left_count),
        "right_count": int(filtered.right_count),
        "environment_count": int(filtered.env_count),
        "stats": filtered.sampling_stats,
    }


def filtered_to_segment_result(filtered: FilteredSegmentResult) -> SegmentResult:
    if filtered.status != "DONE":
        return SegmentResult(
            job=filtered.job,
            status=filtered.status,
            error_code=filtered.error_code,
            error_message=filtered.error_message,
            detections=detections_to_list(filtered.detections),
            sampling=sampling_summary(filtered),
            groups={},
            timings=filtered.timings,
        )
    assert filtered.query_xy is not None
    assert filtered.query_group is not None
    assert filtered.tracks_xy_raw is not None
    assert filtered.tracks_xy_smooth is not None
    assert filtered.visibility is not None
    assert filtered.track_state is not None
    assert filtered.motion_state is not None
    assert filtered.usable is not None

    groups: dict[str, dict[str, Any]] = {}
    group_defs = [
        ("left", 0),
        ("right", 1),
        ("environment", 2),
    ]
    for name, gid in group_defs:
        idx = np.nonzero(filtered.query_group == gid)[0]
        key = GROUP_TO_RESULT_KEY["env"] if name == "environment" else GROUP_TO_RESULT_KEY[name]
        group: dict[str, Any] = {
            "query_xy": filtered.query_xy[idx].astype(np.float32),
            "tracks_xy_raw": filtered.tracks_xy_raw[idx].astype(np.float32),
            "tracks_xy_smooth": filtered.tracks_xy_smooth[idx].astype(np.float32),
            "visibility": filtered.visibility[idx].astype(bool),
            "track_state": filtered.track_state[idx].astype(str),
            "motion_state": filtered.motion_state[idx].astype(str),
            "usable_for_robot_scene_flow": filtered.usable[idx].astype(bool),
        }
        if filtered.confidence is not None:
            group["cotracker_confidence"] = filtered.confidence[idx].astype(np.float32)
        for feat_name, values in filtered.filter_features.items():
            arr = np.asarray(values)
            if arr.shape[:1] == (len(filtered.query_xy),):
                group[feat_name] = arr[idx]
        sampling_features = filtered.sampling_features.get(name, {})
        if isinstance(sampling_features, dict):
            for feat_name, values in sampling_features.items():
                arr = np.asarray(values)
                if arr.shape[:1] == (len(idx),):
                    group[feat_name] = arr
        groups[key] = group

    return SegmentResult(
        job=filtered.job,
        status="DONE",
        detections=detections_to_list(filtered.detections),
        sampling=sampling_summary(filtered),
        groups=groups,
        timings=filtered.timings,
    )
