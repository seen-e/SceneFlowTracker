from scene_flow_tracker.jobs import EpisodeJob
from scene_flow_tracker.segment_planner import plan_segments


def ep(frame_count=38, source_start=0, source_end=38):
    return EpisodeJob(
        dataset="abc",
        episode_id="episode_000",
        episode_index=0,
        task_index=None,
        task=None,
        instruction=None,
        view_key="observation.images.top",
        physical_video_path="/tmp/file.mp4",
        source_start_frame=source_start,
        source_end_frame=source_end,
        manifest_fps=30.0,
        effective_fps=30.0,
        frame_count=frame_count,
    )


def test_segment_boundaries_keep_are_half_open_without_overlap():
    segs = plan_segments(ep(), 15, "keep")
    assert [(s.episode_start_frame, s.episode_end_frame) for s in segs] == [(0, 15), (15, 30), (30, 38)]
    covered = [f for s in segs for f in range(s.episode_start_frame, s.episode_end_frame)]
    assert covered == list(range(38))


def test_tail_drop():
    segs = plan_segments(ep(), 15, "drop")
    assert [(s.episode_start_frame, s.episode_end_frame) for s in segs] == [(0, 15), (15, 30)]


def test_source_offset_mapping():
    segs = plan_segments(ep(frame_count=3885, source_start=3329, source_end=7214), 15, "keep")
    assert (segs[1].episode_start_frame, segs[1].episode_end_frame) == (15, 30)
    assert (segs[1].source_start_frame, segs[1].source_end_frame) == (3344, 3359)
