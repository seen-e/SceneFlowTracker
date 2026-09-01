from __future__ import annotations

import queue
import time
from multiprocessing.queues import Queue

from ..data.types import EndOfStream, FirstFrameItem, YoloBatch
from ..utils.mp import put_with_retry


def yolo_batcher(input_queue: Queue, output_queue: Queue, batch_size: int, flush_sec: float = 0.25) -> None:
    pending: list[FirstFrameItem] = []
    upstream_done = False
    while not upstream_done:
        timeout = flush_sec if pending else None
        try:
            item = input_queue.get(timeout=timeout)
        except queue.Empty:
            item = None
        if isinstance(item, EndOfStream):
            upstream_done = True
        elif item is not None:
            assert isinstance(item, FirstFrameItem)
            pending.append(item)
        if pending and (len(pending) >= batch_size or item is None or upstream_done):
            take = pending[:batch_size]
            pending = pending[batch_size:]
            put_with_retry(output_queue, YoloBatch(items=take, batch_size=len(take), fill_ratio=len(take) / max(1, batch_size)))
    while pending:
        take = pending[:batch_size]
        pending = pending[batch_size:]
        put_with_retry(output_queue, YoloBatch(items=take, batch_size=len(take), fill_ratio=len(take) / max(1, batch_size)))
    put_with_retry(output_queue, EndOfStream(source="yolo_batcher"))
