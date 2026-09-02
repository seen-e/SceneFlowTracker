from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..data.types import CoTrackerBatch, TrackResult
from ..model_parallel import configured_devices
from ..utils.shared_arrays import attach_shared_array, release_shared_array


def patch_cotracker_batch_view(source_root: str | None) -> None:
    if not source_root:
        return
    target = Path(source_root) / "cotracker" / "models" / "core" / "cotracker" / "cotracker3_offline.py"
    if not target.exists():
        return
    text = target.read_text(encoding="utf-8")
    old = "coords_init = coords.view(B * T, N, 2)"
    new = "coords_init = coords.reshape(B * T, N, 2)"
    if old not in text or new in text:
        return
    target.write_text(text.replace(old, new), encoding="utf-8")


def normalize_cotracker_output(pred_tracks: Any, pred_visibility: Any, batch_size: int, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    tracks = pred_tracks.detach().cpu().numpy() if hasattr(pred_tracks, "detach") else np.asarray(pred_tracks)
    vis = pred_visibility.detach().cpu().numpy() if hasattr(pred_visibility, "detach") else np.asarray(pred_visibility)
    if tracks.ndim != 4:
        raise ValueError(f"CoTracker tracks must be rank 4, got {tracks.shape}")
    if tracks.shape[0] != batch_size:
        raise ValueError(f"CoTracker batch mismatch: expected {batch_size}, got {tracks.shape}")
    if tracks.shape[1] == num_points:
        out_tracks = tracks.astype(np.float32, copy=False)
        out_vis = vis.astype(bool, copy=False)
    elif tracks.shape[2] == num_points:
        out_tracks = np.transpose(tracks, (0, 2, 1, 3)).astype(np.float32, copy=False)
        out_vis = np.transpose(vis, (0, 2, 1)).astype(bool, copy=False)
    else:
        raise ValueError(f"Cannot infer CoTracker layout for tracks={tracks.shape}, N={num_points}")
    return out_tracks, out_vis


class CoTrackerModel:
    def __init__(self, cfg: dict[str, Any], device: str | None = None, worker_id: int = 0) -> None:
        import torch

        ccfg = cfg["models"]["cotracker"]
        source_root = ccfg.get("source_root")
        if source_root:
            patch_cotracker_batch_view(str(source_root))
            src = str(Path(source_root))
            if src not in sys.path:
                sys.path.insert(0, src)
        from cotracker.predictor import CoTrackerPredictor

        self.torch = torch
        self.device = self._resolve_device(str(device or configured_devices(ccfg)[0]))
        self.worker_id = int(worker_id)
        self.point_chunk_size = int(ccfg.get("point_chunk_size", ccfg.get("point_batch_size", 1024)))
        self.model = CoTrackerPredictor(checkpoint=str(ccfg["model_path"])).to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        torch = self.torch
        if not requested.startswith("cuda"):
            return requested
        if not torch.cuda.is_available():
            return "cpu"
        try:
            index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        except ValueError:
            index = 0
        if index >= torch.cuda.device_count():
            return "cuda:0"
        return requested

    def track_batch(self, batch: CoTrackerBatch) -> tuple[list[TrackResult], float, float | None]:
        torch = self.torch
        shms = []
        videos = []
        try:
            for item in batch.items:
                shm, arr = attach_shared_array(item.frame_ref)
                shms.append(shm)
                videos.append(np.array(arr, copy=True))
            video_np = np.stack(videos, axis=0)  # [B,T,H,W,C], uint8 RGB
            query_np = np.stack([item.query_xy.astype(np.float32) for item in batch.items], axis=0)
            bsz, t, h, w, _ = video_np.shape
            n = query_np.shape[1]
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            video = torch.from_numpy(video_np).permute(0, 1, 4, 2, 3).float().to(self.device)
            all_tracks = []
            all_vis = []
            started = time.perf_counter()
            with torch.no_grad():
                for start in range(0, n, self.point_chunk_size):
                    end = min(start + self.point_chunk_size, n)
                    queries_xy = torch.from_numpy(query_np[:, start:end]).float().to(self.device)
                    time_zeros = torch.zeros((bsz, end - start, 1), dtype=queries_xy.dtype, device=self.device)
                    queries = torch.cat([time_zeros, queries_xy], dim=-1)
                    pred_tracks, pred_visibility = self.model(video, queries=queries)
                    tracks_np, vis_np = normalize_cotracker_output(pred_tracks, pred_visibility, bsz, end - start)
                    all_tracks.append(tracks_np)
                    all_vis.append(vis_np)
            elapsed = time.perf_counter() - started
            tracks = np.concatenate(all_tracks, axis=1)
            visibility = np.concatenate(all_vis, axis=1)
            peak = None
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                peak = float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
            results = []
            for idx, item in enumerate(batch.items):
                timings = dict(item.timings)
                timings["cotracker_time_sec"] = elapsed / max(1, len(batch.items))
                timings["cotracker_batch_size"] = float(batch.batch_size)
                timings["cotracker_batch_fill_ratio"] = float(batch.fill_ratio)
                timings["cotracker_worker_id"] = float(self.worker_id)
                results.append(
                    TrackResult(
                        job=item.job,
                        tracks_xy=tracks[idx],
                        visibility=visibility[idx],
                        confidence=None,
                        query_xy=item.query_xy,
                        query_group=item.query_group,
                        detections=item.detections,
                        sampling_features=item.sampling_features,
                        sampling_stats=item.sampling_stats,
                        image_height=item.image_height,
                        image_width=item.image_width,
                        timings=timings,
                    )
                )
            return results, elapsed, peak
        finally:
            for shm in shms:
                shm.close()
            for item in batch.items:
                release_shared_array(item.frame_ref, unlink=True)
