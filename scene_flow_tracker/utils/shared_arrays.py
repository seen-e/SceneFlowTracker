from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ..data.types import SharedArrayRef


def create_shared_array(array: np.ndarray, owner: str, debug_id: str) -> SharedArrayRef:
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    shm.close()
    return SharedArrayRef(
        name=shm.name,
        shape=tuple(int(v) for v in arr.shape),
        dtype=str(arr.dtype),
        nbytes=int(arr.nbytes),
        owner=owner,
        debug_id=debug_id,
    )


def attach_shared_array(ref: SharedArrayRef) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=ref.name)
    arr = np.ndarray(tuple(ref.shape), dtype=np.dtype(ref.dtype), buffer=shm.buf)
    return shm, arr


def copy_shared_array(ref: SharedArrayRef) -> np.ndarray:
    shm, arr = attach_shared_array(ref)
    try:
        return np.array(arr, copy=True)
    finally:
        shm.close()


def release_shared_array(ref: SharedArrayRef, unlink: bool = True) -> None:
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=ref.name)
        if unlink:
            shm.unlink()
    except FileNotFoundError:
        return
    finally:
        if shm is not None:
            shm.close()


def shared_ref_to_json(ref: SharedArrayRef) -> dict[str, Any]:
    return {
        "name": ref.name,
        "shape": list(ref.shape),
        "dtype": ref.dtype,
        "nbytes": ref.nbytes,
        "owner": ref.owner,
        "debug_id": ref.debug_id,
    }
