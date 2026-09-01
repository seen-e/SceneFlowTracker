from __future__ import annotations

import queue
from collections import defaultdict
from multiprocessing.queues import Queue

from ..data.types import CoTrackerBatch, DecodedTrackItem, EndOfStream
from ..utils.mp import put_with_retry


def _key(item: DecodedTrackItem) -> tuple[int, int, int, int]:
    shape = item.frame_ref.shape
    return (int(shape[0]), int(shape[1]), int(shape[2]), int(item.query_xy.shape[0]))


def cotracker_batcher(
    input_queue: Queue,
    output_queue: Queue,
    batch_size: int,
    normal_segment_frames: int,
    flush_sec: float = 0.25,
) -> None:
    pending: dict[tuple[int, int, int, int], list[DecodedTrackItem]] = defaultdict(list)
    upstream_done = False
    while not upstream_done:
        try:
            item = input_queue.get(timeout=flush_sec)
        except queue.Empty:
            item = None
        if isinstance(item, EndOfStream):
            upstream_done = True
        elif item is not None:
            assert isinstance(item, DecodedTrackItem)
            key = _key(item)
            if item.job.frame_count != int(normal_segment_frames):
                put_with_retry(
                    output_queue,
                    CoTrackerBatch(items=[item], batch_key=key, batch_size=1, fill_ratio=1.0 / max(1, batch_size), is_tail=True),
                )
            else:
                pending[key].append(item)
                while len(pending[key]) >= batch_size:
                    take = pending[key][:batch_size]
                    pending[key] = pending[key][batch_size:]
                    put_with_retry(
                        output_queue,
                        CoTrackerBatch(items=take, batch_key=key, batch_size=len(take), fill_ratio=1.0, is_tail=False),
                    )
        if upstream_done:
            break
    for key, items in list(pending.items()):
        while items:
            take = items[:batch_size]
            items = items[batch_size:]
            put_with_retry(
                output_queue,
                CoTrackerBatch(items=take, batch_key=key, batch_size=len(take), fill_ratio=len(take) / max(1, batch_size), is_tail=False),
            )
    put_with_retry(output_queue, EndOfStream(source="cotracker_batcher"))
