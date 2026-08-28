from __future__ import annotations

from .jobs import EpisodeJob, SegmentJob


def plan_segments(episode: EpisodeJob, segment_frames: int, tail_policy: str = "keep") -> list[SegmentJob]:
    if segment_frames <= 0:
        raise ValueError("segment_frames must be > 0")
    if tail_policy not in {"keep", "drop"}:
        raise ValueError("tail_policy must be keep or drop")
    segments: list[SegmentJob] = []
    start = 0
    while start < episode.frame_count:
        end = min(start + segment_frames, episode.frame_count)
        if end - start < segment_frames and tail_policy == "drop":
            break
        source_start = episode.source_start_frame + start
        source_end = episode.source_start_frame + end
        segments.append(
            SegmentJob(
                dataset=episode.dataset,
                episode_id=episode.episode_id,
                episode_index=episode.episode_index,
                view_key=episode.view_key,
                physical_video_path=episode.physical_video_path,
                segment_id=len(segments),
                episode_start_frame=start,
                episode_end_frame=end,
                source_start_frame=source_start,
                source_end_frame=source_end,
                frame_count=end - start,
                manifest_fps=episode.manifest_fps,
                effective_fps=episode.effective_fps,
            )
        )
        start = end
    return segments


def plan_all(episodes: list[EpisodeJob], segment_frames: int, tail_policy: str) -> dict[str, list[SegmentJob]]:
    return {ep.episode_id: plan_segments(ep, segment_frames, tail_policy) for ep in episodes}
