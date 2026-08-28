import numpy as np

from scene_flow_tracker.jobs import EpisodeJob, SegmentJob, SegmentResult
from scene_flow_tracker.pipeline.episode_aggregator import EpisodeAggregator
from scene_flow_tracker.storage.reader import EpisodeSceneTrackReader
from scene_flow_tracker.storage.schema import MotionState, SCHEMA_VERSION, TrackState
from scene_flow_tracker.storage.writers import output_npz_path, output_summary_path, write_episode_outputs


def make_episode(idx, frame_count=30, source_start=None):
    source_start = idx * 100 if source_start is None else source_start
    return EpisodeJob("abc", f"ep{idx}", idx, 1, "task", "instruction", "observation.images.top", "/tmp/a.mp4", source_start, source_start + frame_count, 30, 30, frame_count)


def make_segment(ep, sid, start, length):
    return SegmentJob(ep.dataset, ep.episode_id, ep.episode_index, ep.view_key, ep.physical_video_path, sid, start, start + length, ep.source_start_frame + start, ep.source_start_frame + start + length, length, ep.manifest_fps, ep.effective_fps)


def make_group(n, t, track_state="valid", motion_state="moving", usable=True):
    query = np.stack([np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) + 10], axis=1) if n else np.empty((0, 2), np.float32)
    tracks = np.zeros((n, t, 2), np.float32)
    for i in range(n):
        tracks[i, :, 0] = query[i, 0] + np.arange(t)
        tracks[i, :, 1] = query[i, 1]
    return {
        "query_xy": query,
        "tracks_xy_raw": tracks,
        "tracks_xy_smooth": tracks + 0.5,
        "visibility": np.ones((n, t), bool),
        "track_state": np.asarray([track_state] * n),
        "motion_state": np.asarray([motion_state] * n),
        "legacy_motion_state": np.asarray([motion_state] * n),
        "usable_for_robot_scene_flow": np.asarray([usable] * n, dtype=bool),
        "motion_features": [{"visibility_ratio": 1.0, "net_displacement_px": float(t - 1), "path_length_px": float(t - 1), "jitter_rms_px": 0.0} for _ in range(n)],
    }


def result(ep, sid, start=None, length=15, left_n=0, right_n=0, env_n=0, left_state=("valid", "moving", True)):
    start = sid * 15 if start is None else start
    seg = make_segment(ep, sid, start, length)
    detections = [
        {"slot": "left", "bbox_xyxy": [1, 2, 10, 20], "confidence": 0.9},
        {"slot": "right", "bbox_xyxy": [30, 2, 50, 20], "confidence": 0.8},
    ]
    sampling = {
        "query_group_id": np.empty((0,), np.int16),
        "query_local_point_id": np.empty((0,), np.int32),
        "left_bbox": [1, 2, 10, 20],
        "right_bbox": [30, 2, 50, 20],
        "image_width": 64,
        "image_height": 48,
    }
    groups = {
        "left": make_group(left_n, length, *left_state),
        "right": make_group(right_n, length, "valid", "moving", True),
        "environment": make_group(env_n, length, "valid", "static", True),
    }
    return SegmentResult(seg, "DONE", detections=detections, sampling=sampling, groups=groups)


def test_aggregator_sorts_async_results_and_keeps_episodes_separate(tmp_path):
    ep0, ep1 = make_episode(0), make_episode(1)
    agg = EpisodeAggregator([ep0, ep1], {"ep0": 2, "ep1": 1}, tmp_path)
    assert agg.add(result(ep0, 1)) is None
    assert agg.add(result(ep1, 0)) is not None
    assert agg.add(result(ep0, 0)) is not None
    assert output_summary_path(tmp_path, ep0).exists()
    assert output_summary_path(tmp_path, ep1).exists()


def test_schema_1p2_writer_reader_roundtrip_with_padding_and_states(tmp_path):
    ep = make_episode(9, frame_count=38, source_start=3329)
    results = [
        result(ep, 1, start=15, length=15, left_n=100, right_n=61, env_n=276, left_state=("valid", "jitter", False)),
        result(ep, 0, start=0, length=15, left_n=83, right_n=95, env_n=300, left_state=("valid", "static", True)),
        result(ep, 2, start=30, length=8, left_n=0, right_n=12, env_n=20),
    ]
    summary = write_episode_outputs(tmp_path, ep, results)
    path = output_npz_path(tmp_path, ep)
    assert path.name == "observation.images.top_scene_tracks.npz"
    assert summary["schema_version"] == SCHEMA_VERSION
    with np.load(path, allow_pickle=False) as data:
        assert str(data["schema_version"].item()) == SCHEMA_VERSION
        assert data["segment_lengths"].tolist() == [15, 15, 8]
        assert data["left_tracks_raw"].shape == (3, 15, 100, 2)
        assert data["right_tracks_raw"].shape == (3, 15, 95, 2)
        assert data["env_tracks_raw"].shape == (3, 15, 300, 2)
        assert data["source_start_frames"].tolist() == [3329, 3344, 3359]
        assert data["source_frame_indices"][1, :15].tolist() == list(range(3344, 3359))
        assert data["episode_frame_indices"][0, 14] == 14
        assert data["episode_frame_indices"][1, 0] == 15
        assert data["left_num_points"].tolist() == [83, 100, 0]
        assert not data["left_point_valid"][2].any()
        assert np.isnan(data["left_query_xy"][2]).all()
        assert np.isnan(data["left_tracks_raw"][2, 8:]).all()
        assert int(data["left_track_state"][0, 0]) == int(TrackState.VALID)
        assert int(data["left_motion_state"][0, 0]) == int(MotionState.STATIC)
        assert bool(data["left_usable"][0, 0])
        assert int(data["left_motion_state"][1, 0]) == int(MotionState.JITTER)
        assert not bool(data["left_usable"][1, 0])
        assert data["yolo_bboxes"].shape == (3, 1, 2, 4)
        assert data["yolo_frame_indices"][0, 0] == 3329
    with EpisodeSceneTrackReader(path) as reader:
        assert reader.num_segments == 3
        left = reader.get_left(1)
        assert left["query_xy"].shape == (100, 2)
        assert left["tracks_raw"].shape == (15, 100, 2)
        assert left["motion_state"][0] == int(MotionState.JITTER)
        tail = reader.get_right(2)
        assert tail["tracks_raw"].shape == (8, 12, 2)
