from __future__ import annotations

import logging
from multiprocessing.queues import Queue

from ..data.types import CoTrackerBatch, EndOfStream
from ..inference.cotracker_model import CoTrackerModel
from ..utils.mp import configure_process_logging, put_with_retry


def cotracker_worker(worker_id: int, input_queue: Queue, output_queue: Queue, cfg: dict, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("cotracker_worker[%s] loading model", worker_id)
    model = CoTrackerModel(cfg)
    logging.info("cotracker_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, CoTrackerBatch)
        results, _elapsed, peak = model.track_batch(item)
        for result in results:
            if peak is not None:
                result.timings["cotracker_gpu_peak_memory_mb"] = peak
            put_with_retry(output_queue, result)
