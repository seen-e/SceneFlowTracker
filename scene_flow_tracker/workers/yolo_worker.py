from __future__ import annotations

import logging
from multiprocessing.queues import Queue

from ..data.types import EndOfStream, YoloBatch
from ..inference.yolo_model import YoloModel
from ..utils.mp import configure_process_logging, put_with_retry


def yolo_worker(worker_id: int, input_queue: Queue, output_queue: Queue, cfg: dict, log_level: int = logging.INFO) -> None:
    configure_process_logging(log_level)
    logging.info("yolo_worker[%s] loading model", worker_id)
    model = YoloModel(cfg)
    logging.info("yolo_worker[%s] started", worker_id)
    while True:
        item = input_queue.get()
        if isinstance(item, EndOfStream):
            put_with_retry(output_queue, item)
            return
        assert isinstance(item, YoloBatch)
        results, elapsed = model.predict_batch(item.items)
        for result in results:
            result.timings["yolo_batch_size"] = float(item.batch_size)
            result.timings["yolo_batch_fill_ratio"] = float(item.fill_ratio)
            result.timings["yolo_forward_time_sec"] = elapsed
            put_with_retry(output_queue, result)
