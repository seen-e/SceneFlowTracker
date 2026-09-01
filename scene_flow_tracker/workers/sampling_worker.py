from __future__ import annotations

import logging
import time
from multiprocessing.queues import Queue

import numpy as np

from ..algorithms.environment_sampling import sample_environment_points
from ..algorithms.query_allocator import allocate_initial_queries, final_environment_target
from ..algorithms.query_builder import build_query_set, stable_seed
from ..algorithms.robot_sampling import sample_robot_points
from ..data.types import EndOfStream, SamplingResult, YoloDetectionResult
from ..pipeline.result_builder import filtered_failure
from ..utils.mp import configure_process_logging, put_with_retry


def _valid_bboxes(det: YoloDetectionResult) -> list[np.ndarray]:
    out = []
    if det.left_bbox_valid and det.left_bbox_xyxy is not None:
        out.append(det.left_bbox_xyxy)
    if det.right_bbox_valid and det.right_bbox_xyxy is not None:
        out.append(det.right_bbox_xyxy)
    return out


def sample_queries(det: YoloDetectionResult, cfg: dict) -> SamplingResult:
    started = time.perf_counter()
    qcfg = (cfg.get("sampling", {}) or {}).get("query_allocation", {})
    total = int(qcfg.get("total_query_points", 300))
    per_arm = int(qcfg.get("points_per_detected_arm", 100))
    allocation = allocate_initial_queries(det.left_bbox_valid, det.right_bbox_valid, total, per_arm)
    left_target = allocation["left"]
    right_target = allocation["right"]
    seed = stable_seed(det.job, int((cfg.get("sampling", {}) or {}).get("seed", 0)))
    robot_cfg = cfg.get("robot_sampling", {}) or {}
    env_cfg = cfg.get("environment_sampling", {}) or {}
    left = sample_robot_points(det.first_frame_rgb, det.left_bbox_xyxy, left_target, robot_cfg, seed=seed + 11)
    right = sample_robot_points(det.first_frame_rgb, det.right_bbox_xyxy, right_target, robot_cfg, seed=seed + 23)
    left_pts = left["points_xy"]
    right_pts = right["points_xy"]
    env_target = final_environment_target(total, len(left_pts), len(right_pts))
    exclude = np.concatenate([left_pts, right_pts], axis=0) if len(left_pts) + len(right_pts) else None
    env = sample_environment_points(det.first_frame_rgb, _valid_bboxes(det), env_target, env_cfg, seed=seed + 37, exclude_points=exclude)
    query_xy, query_group, _layout = build_query_set(left_pts, right_pts, env["points_xy"], total, det.image_width, det.image_height)
    timings = dict(det.timings)
    timings["sampling_time_sec"] = time.perf_counter() - started
    return SamplingResult(
        job=det.job,
        detections=det,
        query_xy=query_xy,
        query_group=query_group,
        left_count=len(left_pts),
        right_count=len(right_pts),
        env_count=len(env["points_xy"]),
        sampling_features={"left": left["features"], "right": right["features"], "environment": env["features"]},
        sampling_stats={"left": left["stats"], "right": right["stats"], "environment": env["stats"], "total_query_points": total},
        timings=timings,
    )


def sampling_worker(worker_id: int, input_queue: Queue, output_queue: Queue, final_queue: Queue, cfg: dict, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("sampling_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, YoloDetectionResult)
        try:
            put_with_retry(output_queue, sample_queries(item, cfg))
        except Exception as exc:
            logging.exception("sampling failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            put_with_retry(final_queue, filtered_failure(item.job, type(exc).__name__, str(exc), dict(item.timings)))
