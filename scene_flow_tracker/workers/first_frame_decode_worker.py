from __future__ import annotations

import logging
import time
from multiprocessing.queues import Queue

from ..data.types import EndOfStream, FirstFrameItem
from ..jobs import SegmentJob
from ..pipeline.result_builder import filtered_failure
from ..utils.mp import configure_process_logging, put_with_retry
from ..video_decode import decode_first_frame_rgb


def decode_first_frame(job: SegmentJob):
    return decode_first_frame_rgb(job.physical_video_path, job.source_start_frame)


def first_frame_decode_worker(worker_id: int, input_queue: Queue, output_queue: Queue, final_queue: Queue, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("first_frame_decode_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, SegmentJob)
        started = time.perf_counter()
        try:
            frame_rgb = decode_first_frame(item)
            put_with_retry(output_queue, FirstFrameItem(job=item, frame_rgb=frame_rgb, timings={"first_frame_decode_time_sec": time.perf_counter() - started}))
        except Exception as exc:
            logging.exception("first frame decode failed episode=%s segment=%s", item.episode_id, item.segment_id)
            put_with_retry(final_queue, filtered_failure(item, type(exc).__name__, str(exc), {"first_frame_decode_time_sec": time.perf_counter() - started}))
