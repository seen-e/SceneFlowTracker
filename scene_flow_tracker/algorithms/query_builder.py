from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..jobs import SegmentJob


LEFT_GROUP = 0
RIGHT_GROUP = 1
ENV_GROUP = 2


@dataclass(frozen=True)
class QueryLayout:
    group_id: np.ndarray
    local_point_id: np.ndarray
    slices: dict[str, slice]


def stable_seed(job: SegmentJob, base_seed: int = 0) -> int:
    text = f"{base_seed}|{job.dataset}|{job.episode_id}|{job.segment_id}|{job.view_key}"
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _unique_rows(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    rounded = np.round(points).astype(np.int64)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx.sort()
    return points[idx].astype(np.float32)


def assert_no_duplicates(points: np.ndarray) -> None:
    rounded = np.round(points).astype(np.int64)
    if len(np.unique(rounded, axis=0)) != len(rounded):
        raise ValueError("query points contain duplicates")


def remove_existing(points: np.ndarray, existing: Iterable[tuple[int, int]]) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    used = set(existing)
    keep = []
    for p in _unique_rows(points):
        key = (int(round(float(p[0]))), int(round(float(p[1]))))
        if key not in used:
            keep.append(p)
            used.add(key)
    return np.asarray(keep, dtype=np.float32).reshape(-1, 2)


def build_query_set(left: np.ndarray, right: np.ndarray, env: np.ndarray, total_query_points: int, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray, QueryLayout]:
    left = _unique_rows(np.asarray(left, dtype=np.float32).reshape(-1, 2))
    right = remove_existing(np.asarray(right, dtype=np.float32).reshape(-1, 2), [(int(round(x)), int(round(y))) for x, y in left])
    used = [(int(round(x)), int(round(y))) for x, y in np.concatenate([left, right], axis=0)] if len(left) + len(right) else []
    env = remove_existing(np.asarray(env, dtype=np.float32).reshape(-1, 2), used)
    query_xy = np.concatenate([left, right, env], axis=0).astype(np.float32)
    if len(query_xy) != int(total_query_points):
        raise ValueError(f"query count mismatch: expected={total_query_points} got={len(query_xy)}")
    if np.any(query_xy[:, 0] < 0) or np.any(query_xy[:, 0] >= image_width) or np.any(query_xy[:, 1] < 0) or np.any(query_xy[:, 1] >= image_height):
        raise ValueError("query point out of image bounds")
    assert_no_duplicates(query_xy)
    group = np.concatenate([
        np.full((len(left),), LEFT_GROUP, dtype=np.int16),
        np.full((len(right),), RIGHT_GROUP, dtype=np.int16),
        np.full((len(env),), ENV_GROUP, dtype=np.int16),
    ])
    local = np.concatenate([
        np.arange(len(left), dtype=np.int32),
        np.arange(len(right), dtype=np.int32),
        np.arange(len(env), dtype=np.int32),
    ])
    layout = QueryLayout(
        group_id=group,
        local_point_id=local,
        slices={
            "left": slice(0, len(left)),
            "right": slice(len(left), len(left) + len(right)),
            "environment": slice(len(left) + len(right), len(query_xy)),
        },
    )
    return query_xy, group, layout
