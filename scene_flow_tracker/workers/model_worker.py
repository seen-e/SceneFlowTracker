from __future__ import annotations

import logging
from multiprocessing.queues import Queue

from ..jobs import DecodedSegment
from ..pipeline.model_loader import load_model_bundle
from ..pipeline.segment_processor import process_segment


def model_worker(worker_id: int, cfg: dict, device_name: str, decoded_queue: Queue, result_queue: Queue) -> None:
    logging.info("model_worker[%s] loading models on %s", worker_id, device_name)
    bundle = load_model_bundle(cfg, device_name)
    logging.info("model_worker[%s] ready", worker_id)
    while True:
        item = decoded_queue.get()
        if item is None:
            result_queue.put(None)
            break
        assert isinstance(item, DecodedSegment)
        result = process_segment(item, cfg, bundle)
        result_queue.put(result)
