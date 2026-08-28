from multiprocessing import get_context

from scene_flow_tracker.jobs import DecodedSegment, SegmentJob
from scene_flow_tracker.workers.decode_worker import decode_worker


def test_decode_worker_does_not_emit_model_sentinel(tmp_path):
    ctx = get_context("spawn")
    job_queue = ctx.Queue()
    decoded_queue = ctx.Queue()
    result_queue = ctx.Queue()
    job_queue.put(None)
    decode_worker(0, job_queue, decoded_queue, result_queue)
    assert decoded_queue.empty()
