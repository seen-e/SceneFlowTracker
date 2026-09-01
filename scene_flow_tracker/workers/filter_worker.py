from __future__ import annotations

import logging
import time
from multiprocessing.queues import Queue

import numpy as np

from ..algorithms.trajectory_filter import filter_tracks
from ..data.types import EndOfStream, FilteredSegmentResult, TrackResult
from ..utils.mp import configure_process_logging, put_with_retry


def filter_track_result(track: TrackResult, cfg: dict) -> FilteredSegmentResult:
    started = time.perf_counter()
    groups = filter_tracks(track, cfg.get("trajectory_filter", {}) or {})
    track_state = np.concatenate([groups["left"]["track_state"], groups["right"]["track_state"], groups["environment"]["track_state"]], axis=0)
    motion_state = np.concatenate([groups["left"]["motion_state"], groups["right"]["motion_state"], groups["environment"]["motion_state"]], axis=0)
    usable = np.concatenate(
        [
            groups["left"]["usable_for_robot_scene_flow"],
            groups["right"]["usable_for_robot_scene_flow"],
            groups["environment"]["usable_for_robot_scene_flow"],
        ],
        axis=0,
    )
    smooth = np.concatenate([groups["left"]["tracks_xy_smooth"], groups["right"]["tracks_xy_smooth"], groups["environment"]["tracks_xy_smooth"]], axis=0)
    filter_features = {}
    for name in [
        "visibility_ratio",
        "net_displacement",
        "path_length",
        "path_efficiency",
        "jitter_rms",
        "jitter_residual_ratio",
        "turn_consistency",
        "turn_angle_mad",
        "normalized_jerk",
        "direction_reversal_ratio",
    ]:
        vals = [groups[g].get(name) for g in ("left", "right", "environment")]
        if all(v is not None for v in vals):
            filter_features[name] = np.concatenate(vals, axis=0).astype(np.float32)
    timings = dict(track.timings)
    timings["filter_time_sec"] = time.perf_counter() - started
    return FilteredSegmentResult(
        job=track.job,
        status="DONE",
        detections=track.detections,
        query_xy=track.query_xy,
        query_group=track.query_group,
        left_count=int(np.count_nonzero(track.query_group == 0)),
        right_count=int(np.count_nonzero(track.query_group == 1)),
        env_count=int(np.count_nonzero(track.query_group == 2)),
        sampling_features=track.sampling_features,
        sampling_stats=track.sampling_stats,
        tracks_xy_raw=track.tracks_xy,
        tracks_xy_smooth=smooth,
        visibility=track.visibility,
        confidence=track.confidence,
        track_state=track_state,
        motion_state=motion_state,
        usable=usable,
        filter_features=filter_features,
        image_height=track.image_height,
        image_width=track.image_width,
        timings=timings,
    )


def filter_worker(worker_id: int, input_queue: Queue, output_queue: Queue, cfg: dict, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("filter_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, TrackResult)
        try:
            put_with_retry(output_queue, filter_track_result(item, cfg))
        except Exception as exc:
            logging.exception("filter failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            put_with_retry(
                output_queue,
                FilteredSegmentResult(
                    job=item.job,
                    status="FAILED",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                    detections=item.detections,
                    image_height=item.image_height,
                    image_width=item.image_width,
                    timings=dict(item.timings),
                ),
            )
