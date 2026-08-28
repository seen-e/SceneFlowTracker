import cv2
import numpy as np

from scene_flow_tracker.jobs import SegmentJob
from scene_flow_tracker.workers.decode_worker import decode_half_open


def test_decode_half_open_exact_count(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    for i in range(6):
        writer.write(np.full((16, 16, 3), i * 20, np.uint8))
    writer.release()
    job = SegmentJob("abc", "ep", 0, "observation.images.top", str(path), 0, 2, 5, 2, 5, 3, 10.0, 10.0)
    frames = decode_half_open(job)
    assert len(frames) == 3
