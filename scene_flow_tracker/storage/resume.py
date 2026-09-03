from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..jobs import EpisodeJob
from .writers import safe_view_name


EpisodeViewKey = tuple[str, str, str]


@dataclass(frozen=True)
class _ExistingViewCandidate:
    key: EpisodeViewKey
    npz_path: Path
    summary_path: Path


@dataclass
class ResumeScanResult:
    completed: set[EpisodeViewKey] = field(default_factory=set)
    existing_episode_dirs: int = 0
    existing_npz_files: int = 0
    missing_summary_files: int = 0
    selected_existing_dirs: int = 0
    checked_view_outputs: int = 0
    elapsed_sec: float = 0.0

    def is_completed(self, episode: EpisodeJob) -> bool:
        return episode_key(episode) in self.completed


def episode_key(episode: EpisodeJob) -> EpisodeViewKey:
    return (episode.dataset, episode.episode_id, safe_view_name(episode.view_key))


def _candidate_is_completed(candidate: _ExistingViewCandidate, require_summary: bool) -> tuple[EpisodeViewKey, bool, bool]:
    try:
        if candidate.npz_path.stat().st_size <= 0:
            return candidate.key, False, False
    except OSError:
        return candidate.key, False, False
    if not require_summary:
        return candidate.key, True, False
    try:
        if candidate.summary_path.stat().st_size <= 0:
            return candidate.key, False, True
        summary = json.loads(candidate.summary_path.read_text(encoding="utf-8"))
    except Exception:
        return candidate.key, False, True
    return candidate.key, int(summary.get("segments_failed", 0)) == 0, False


def scan_completed_episode_views(output_root: Path, episodes: list[EpisodeJob], *, require_summary: bool = True, workers: int = 1) -> ResumeScanResult:
    """Scan output_root once and return completed episode/view outputs.

    The runner may receive hundreds of thousands of manifest entries. Calling
    stat on every expected NPZ and summary pair creates a slow metadata storm on
    network filesystems, so this follows the existing-output census pattern:
    list existing output dirs first, then only inspect view outputs that already
    exist under output_root. Summary validation is parallelized because network
    filesystem metadata and small JSON reads dominate resume-scan time.
    """
    started = time.perf_counter()
    result = ResumeScanResult()
    if not output_root.exists():
        result.elapsed_sec = time.perf_counter() - started
        return result

    selected_keys = {episode_key(ep) for ep in episodes}
    selected_episode_dirs = {(dataset, episode_id) for dataset, episode_id, _view in selected_keys}
    selected_datasets = {dataset for dataset, _episode_id, _view in selected_keys}
    candidates: list[_ExistingViewCandidate] = []

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
                view = npz_path.name[: -len("_scene_tracks.npz")]
                key = (dataset, episode_id, view)
                if key not in selected_keys:
                    continue
                candidates.append(_ExistingViewCandidate(key=key, npz_path=npz_path, summary_path=episode_dir / f"{view}_summary.json"))

    result.checked_view_outputs = len(candidates)
    workers = max(1, int(workers))
    if workers == 1 or len(candidates) <= 1:
        checked = [_candidate_is_completed(candidate, require_summary) for candidate in candidates]
    else:
        checked = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="resume-scan") as pool:
            futures = [pool.submit(_candidate_is_completed, candidate, require_summary) for candidate in candidates]
            for fut in as_completed(futures):
                checked.append(fut.result())
    for key, completed, missing_summary in checked:
        if missing_summary:
            result.missing_summary_files += 1
        if completed:
            result.completed.add(key)
    result.elapsed_sec = time.perf_counter() - started
    return result
