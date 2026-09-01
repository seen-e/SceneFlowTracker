from __future__ import annotations

import numpy as np


def smooth_tracks(tracks_xy: np.ndarray, cfg: dict) -> np.ndarray:
    tracks = np.asarray(tracks_xy, dtype=np.float32)
    if not bool(cfg.get("enabled", True)) or tracks.shape[1] < 3:
        return tracks.copy()
    window = int(cfg.get("window_size", cfg.get("window_length", 5)))
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    window = min(window, tracks.shape[1] if tracks.shape[1] % 2 == 1 else tracks.shape[1] - 1)
    if window < 3:
        return tracks.copy()
    pad = window // 2
    padded = np.pad(tracks, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(tracks)
    for t in range(tracks.shape[1]):
        out[:, t] = padded[:, t : t + window].mean(axis=1)
    return out
