from __future__ import annotations

from multiprocessing import get_context


def make_context(method: str = "spawn"):
    return get_context(method)


def make_queues(ctx, cfg: dict):
    qcfg = cfg["queues"]
    return (
        ctx.Queue(maxsize=int(qcfg["segment_job_queue_size"])),
        ctx.Queue(maxsize=int(qcfg["decoded_segment_queue_size"])),
        ctx.Queue(maxsize=int(qcfg["result_queue_size"])),
    )
