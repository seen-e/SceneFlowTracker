from __future__ import annotations

import logging
import queue
import time
from multiprocessing.queues import Queue

import cv2

from ..jobs import DecodedSegment, SegmentJob, SegmentResult


def decode_half_open(job: SegmentJob) -> list:
    cap = cv2.VideoCapture(job.physical_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(job.physical_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, job.source_start_frame)
    frames = []
    for _ in range(job.frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != job.frame_count:
        raise RuntimeError(f"DECODE_FRAME_COUNT_MISMATCH expected={job.frame_count} got={len(frames)}")
    return frames


def decode_worker(worker_id: int, job_queue: Queue, decoded_queue: Queue, result_queue: Queue, continue_on_error: bool = True) -> None:
    logging.info("decode_worker[%s] started", worker_id)
    while True:
        try:
            job = job_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if job is None:
            break
        assert isinstance(job, SegmentJob)
        started = time.perf_counter()
        try:
            frames = decode_half_open(job)
            decoded_queue.put(DecodedSegment(job=job, frames_bgr=frames, decode_time_sec=time.perf_counter() - started))
        except Exception as exc:
            logging.exception("decode_worker[%s] failed episode=%s segment=%s", worker_id, job.episode_id, job.segment_id)
            result_queue.put(
                SegmentResult(
                    job=job,
                    status="FAILED",
                    error_code=type(exc).__name__ if str(exc).split()[0] != "DECODE_FRAME_COUNT_MISMATCH" else "DECODE_FRAME_COUNT_MISMATCH",
                    error_message=str(exc),
                    timings={"decode_time_sec": time.perf_counter() - started},
                )
            )
            if not continue_on_error:
                break
