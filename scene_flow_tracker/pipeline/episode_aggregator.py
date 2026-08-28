from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..jobs import EpisodeJob, SegmentResult
from ..storage.writers import append_processing_manifest, write_episode_outputs


class EpisodeAggregator:
    def __init__(self, episodes: list[EpisodeJob], expected_counts: dict[str, int], output_root: Path, atomic_write: bool = True, cfg: dict | None = None) -> None:
        self.episodes = {ep.episode_id: ep for ep in episodes}
        self.expected_counts = expected_counts
        self.output_root = output_root
        self.atomic_write = atomic_write
        self.cfg = cfg or {}
        self.results: dict[str, list[SegmentResult]] = defaultdict(list)
        self.done: set[str] = set()
        self.manifest_path = output_root / "processing_manifest.jsonl"

    def add(self, result: SegmentResult) -> dict | None:
        episode_id = result.job.episode_id
        self.results[episode_id].append(result)
        append_processing_manifest(
            self.manifest_path,
            {
                "dataset": result.job.dataset,
                "episode_id": episode_id,
                "episode_index": result.job.episode_index,
                "view": result.job.view_key,
                "video_path": result.job.physical_video_path,
                "source_start_frame": result.job.source_start_frame,
                "source_end_frame": result.job.source_end_frame,
                "segment_id": result.job.segment_id,
                "status": result.status,
                "error": result.error_message,
            },
        )
        if episode_id not in self.done and len(self.results[episode_id]) >= self.expected_counts[episode_id]:
            self.done.add(episode_id)
            failed = [r for r in self.results[episode_id] if r.status != "DONE"]
            summary = write_episode_outputs(self.output_root, self.episodes[episode_id], self.results[episode_id], self.atomic_write, self.cfg)
            append_processing_manifest(
                self.manifest_path,
                {
                    "dataset": result.job.dataset,
                    "episode_id": episode_id,
                    "episode_index": result.job.episode_index,
                    "view": result.job.view_key,
                    "video_path": result.job.physical_video_path,
                    "source_start_frame": self.episodes[episode_id].source_start_frame,
                    "source_end_frame": self.episodes[episode_id].source_end_frame,
                    "status": "DONE" if not failed else "FAILED",
                    "num_segments": self.expected_counts[episode_id],
                    "finished_segments": self.expected_counts[episode_id] - len(failed),
                    "failed_segments": len(failed),
                    "output_path": summary["output_npz"],
                },
            )
            return summary
        return None
