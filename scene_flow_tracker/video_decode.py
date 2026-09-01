from __future__ import annotations

import json
import subprocess
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=512)
def probe_video_size(path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    payload = json.loads(proc.stdout.decode("utf-8"))
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def decode_frames_ffmpeg(path: str, start_frame: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    width, height = probe_video_size(path)
    end_frame = int(start_frame) + int(frame_count) - 1
    vf = f"select='between(n\\,{int(start_frame)}\\,{end_frame})',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    expected = int(frame_count) * height * width * 3
    if len(proc.stdout) != expected:
        got = len(proc.stdout) // max(1, height * width * 3)
        raise RuntimeError(f"FFMPEG_FRAME_COUNT_MISMATCH expected={frame_count} got={got}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(int(frame_count), height, width, 3).copy()


def decode_frames_cv2(path: str, start_frame: int, frame_count: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"DECODE_FRAME_COUNT_MISMATCH expected={frame_count} got={len(frames)}")
    return np.stack(frames, axis=0).astype(np.uint8)


def decode_frames_rgb(path: str, start_frame: int, frame_count: int) -> np.ndarray:
    try:
        return decode_frames_ffmpeg(path, start_frame, frame_count)
    except Exception:
        return decode_frames_cv2(path, start_frame, frame_count)


def decode_first_frame_rgb(path: str, frame_index: int) -> np.ndarray:
    return decode_frames_rgb(path, frame_index, 1)[0]
