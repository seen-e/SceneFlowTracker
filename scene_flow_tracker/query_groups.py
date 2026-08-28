from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUP_ID = {"left": 0, "right": 1, "environment": 2}
GROUP_NAME = {v: k for k, v in GROUP_ID.items()}


@dataclass(frozen=True)
class QueryLayout:
    counts: dict[str, int]
    slices: dict[str, slice]
    group_id: np.ndarray
    local_point_id: np.ndarray


def merge_queries(groups: dict[str, np.ndarray]) -> tuple[np.ndarray, QueryLayout]:
    ordered = ["left", "right", "environment"]
    arrays = []
    counts: dict[str, int] = {}
    slices: dict[str, slice] = {}
    group_ids = []
    local_ids = []
    cursor = 0
    for name in ordered:
        arr = np.asarray(groups.get(name, np.empty((0, 2), np.float32)), dtype=np.float32)
        if arr.size == 0:
            arr = np.empty((0, 2), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} queries must have shape (N,2), got {arr.shape}")
        n = int(arr.shape[0])
        counts[name] = n
        slices[name] = slice(cursor, cursor + n)
        cursor += n
        arrays.append(arr)
        group_ids.extend([GROUP_ID[name]] * n)
        local_ids.extend(range(n))
    merged = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 2), dtype=np.float32)
    return merged, QueryLayout(counts, slices, np.asarray(group_ids, np.int16), np.asarray(local_ids, np.int32))


def split_by_layout(values: np.ndarray, layout: QueryLayout) -> dict[str, np.ndarray]:
    return {name: values[slc] for name, slc in layout.slices.items()}
