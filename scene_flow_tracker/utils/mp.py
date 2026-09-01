from __future__ import annotations

import logging
import queue
import time
from multiprocessing.queues import Queue
from typing import Any


def configure_process_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        force=True,
    )


def put_with_retry(q: Queue, item: Any, timeout: float = 0.5) -> None:
    while True:
        try:
            q.put(item, timeout=timeout)
            return
        except queue.Full:
            time.sleep(0.01)
