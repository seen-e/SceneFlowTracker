from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..jobs import EpisodeJob
from .writers import safe_view_name


EpisodeViewKey = tuple[str, str, str]


@dataclass
class ResumeScanResult:
    completed: set[EpisodeViewKey] = field(default_factory=set)
    existing_episode_dirs: int = 0
    existing_npz_files: int = 0
    missing_summary_files: int = 0
    selected_existing_dirs: int = 0
    elapsed_sec: float = 0.0

    def is_completed(self, episode: EpisodeJob) -> bool:
        return episode_key(episode) in self.completed


def episode_key(episode: EpisodeJob) -> EpisodeViewKey:
    return (episode.dataset, episode.episode_id, safe_view_name(episode.view_key))


def scan_completed_episode_views(output_root: Path, episodes: list[EpisodeJob], *, require_summary: bool = True) -> ResumeScanResult:
    """Scan output_root once and return completed episode/view outputs.

    The runner may receive hundreds of thousands of manifest entries. Calling
    stat on every expected NPZ and summary pair creates a slow metadata storm on
    network filesystems, so this follows the existing-output census pattern:
    list existing output dirs first, then only inspect dirs that overlap the
    selected episode/view keys.
    """
    started = time.perf_counter()
    result = ResumeScanResult()
    if not output_root.exists():
        result.elapsed_sec = time.perf_counter() - started
        return result

    selected_keys = {episode_key(ep) for ep in episodes}
    selected_episode_dirs = {(dataset, episode_id) for dataset, episode_id, _view in selected_keys}
    selected_datasets = {dataset for dataset, _episode_id, _view in selected_keys}

    for dataset_dir in output_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name not in selected_datasets:
            continue
        dataset = dataset_dir.name
        for episode_dir in dataset_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            result.existing_episode_dirs += 1
            episode_id = episode_dir.name
            if (dataset, episode_id) not in selected_episode_dirs:
                continue
            result.selected_existing_dirs += 1
            for npz_path in episode_dir.glob("*_scene_tracks.npz"):
                result.existing_npz_files += 1
                if npz_path.stat().st_size <= 0:
                    continue
                view = npz_path.name[: -len("_scene_tracks.npz")]
                key = (dataset, episode_id, view)
                if key not in selected_keys:
                    continue
                summary_path = episode_dir / f"{view}_summary.json"
                if require_summary and (not summary_path.exists() or summary_path.stat().st_size <= 0):
                    result.missing_summary_files += 1
                    continue
                if require_summary:
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        result.missing_summary_files += 1
                        continue
                    if int(summary.get("segments_failed", 0)) > 0:
                        continue
                result.completed.add(key)
    result.elapsed_sec = time.perf_counter() - started
    return result
