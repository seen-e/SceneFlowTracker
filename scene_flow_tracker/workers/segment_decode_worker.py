from __future__ import annotations

import logging
import time
from multiprocessing.queues import Queue

import numpy as np

from ..data.types import DecodedTrackItem, EndOfStream, SamplingResult
from ..pipeline.result_builder import filtered_failure
from ..utils.mp import configure_process_logging, put_with_retry
from ..utils.shared_arrays import create_shared_array
from ..video_decode import decode_frames_rgb


def decode_segment_rgb(job) -> np.ndarray:
    return decode_frames_rgb(job.physical_video_path, job.source_start_frame, job.frame_count)


def segment_decode_worker(worker_id: int, input_queue: Queue, output_queue: Queue, final_queue: Queue, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("segment_decode_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, SamplingResult)
        started = time.perf_counter()
        try:
            frames = decode_segment_rgb(item.job)
            ref = create_shared_array(frames, owner=f"segment_decode_worker_{worker_id}", debug_id=f"{item.job.episode_id}:{item.job.segment_id}")
            timings = dict(item.timings)
            timings["segment_decode_time_sec"] = time.perf_counter() - started
            put_with_retry(
                output_queue,
                DecodedTrackItem(
                    job=item.job,
                    frame_ref=ref,
                    query_xy=item.query_xy,
                    query_group=item.query_group,
                    detections=item.detections,
                    sampling_features=item.sampling_features,
                    sampling_stats=item.sampling_stats,
                    image_height=item.detections.image_height,
                    image_width=item.detections.image_width,
                    timings=timings,
                ),
            )
        except Exception as exc:
            logging.exception("segment decode failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            put_with_retry(final_queue, filtered_failure(item.job, type(exc).__name__, str(exc), dict(item.timings)))
