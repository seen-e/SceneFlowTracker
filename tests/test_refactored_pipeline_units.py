import numpy as np

from scene_flow_tracker.algorithms.query_allocator import allocate_initial_queries, final_environment_target
from scene_flow_tracker.algorithms.query_builder import build_query_set
from scene_flow_tracker.data.types import DecodedTrackItem, SharedArrayRef, TrackResult, YoloDetectionResult
from scene_flow_tracker.inference.cotracker_model import normalize_cotracker_output
from scene_flow_tracker.inference.yolo_model import assign_arm_slots
from scene_flow_tracker.jobs import EpisodeJob, SegmentJob
from scene_flow_tracker.orchestration.pipeline_runner import PipelineRunner
from scene_flow_tracker.pipeline.result_builder import filtered_to_segment_result
from scene_flow_tracker.storage.segment_cache import read_segment_cache, validate_segment_cache, write_segment_cache
from scene_flow_tracker.utils.shared_arrays import copy_shared_array, create_shared_array, release_shared_array
from scene_flow_tracker.workers.filter_worker import filter_track_result
from scene_flow_tracker.workers.sampling_worker import sample_queries


def make_segment(segment_id=0, frame_count=4):
    return SegmentJob(
        dataset="abc",
        episode_id="ep0",
        episode_index=0,
        view_key="observation.images.top",
        physical_video_path="/tmp/video.mp4",
        segment_id=segment_id,
        episode_start_frame=segment_id * frame_count,
        episode_end_frame=(segment_id + 1) * frame_count,
        source_start_frame=segment_id * frame_count,
        source_end_frame=(segment_id + 1) * frame_count,
        frame_count=frame_count,
        manifest_fps=30.0,
        effective_fps=30.0,
    )


def make_episode():
    return EpisodeJob("abc", "ep0", 0, None, None, None, "observation.images.top", "/tmp/video.mp4", 0, 12, 30.0, 30.0, 12)


def make_cfg(tmp_path):
    return {
        "video": {"segment_frames": 4, "tail_policy": "keep"},
        "workers": {
            "first_frame_decode_workers": 1,
            "sampling_workers": 1,
            "segment_decode_workers": 1,
            "filter_workers": 1,
        },
        "pipeline": {"max_inflight_segments": 8},
        "models": {
            "yolo": {"model_path": str(tmp_path / "yolo.pt"), "device": "cpu", "batch_size": 2, "imgsz": 64, "conf": 0.1, "iou": 0.7},
            "cotracker": {"model_path": str(tmp_path / "cot.pth"), "device": "cpu", "segment_batch_size": 2, "point_chunk_size": 32},
        },
        "sampling": {"seed": 0, "query_allocation": {"total_query_points": 30, "points_per_detected_arm": 10}},
        "robot_sampling": {},
        "environment_sampling": {"enabled": True},
        "trajectory_filter": {
            "tracking_validity": {"enabled": False},
            "smoothing": {"enabled": False},
            "motion_classification": {"static": {}, "moving": {}, "jitter_v2": {}, "structured_motion": {}},
            "output_policy": {"keep_static": True, "keep_moving": True, "keep_jitter": False, "keep_uncertain": True, "keep_partial_tracks": True},
        },
        "cache": {"enabled": True, "dirname": ".segment_cache", "delete_after_successful_merge": False, "retry_cached_failed_segments": False},
        "batch": {"atomic_write": True, "resume": False, "group_by_physical_video": True, "continue_on_segment_error": True},
        "output": {"output_root": str(tmp_path), "compression": "stored", "save_sampling_features": True, "save_filter_features": True, "save_cotracker_confidence": True},
    }


def test_query_allocation_transfers_missing_robot_quota_to_environment():
    assert allocate_initial_queries(False, False, 300, 100) == {"left": 0, "right": 0, "environment": 300}
    assert allocate_initial_queries(True, False, 300, 100) == {"left": 100, "right": 0, "environment": 200}
    assert allocate_initial_queries(True, True, 300, 100) == {"left": 100, "right": 100, "environment": 100}
    assert final_environment_target(300, 75, 100) == 125


def test_yolo_assignment_accepts_zero_one_two_bboxes():
    left, lc, right, rc, method = assign_arm_slots([], 100, 80)
    assert left is None and right is None and method == "none"
    left, lc, right, rc, method = assign_arm_slots([{"bbox_xyxy": [5, 5, 20, 20], "confidence": 0.7, "class_id": 0}], 100, 80)
    assert left is not None and right is None
    left, lc, right, rc, method = assign_arm_slots(
        [
            {"bbox_xyxy": [70, 5, 90, 20], "confidence": 0.7, "class_id": 0},
            {"bbox_xyxy": [10, 5, 30, 20], "confidence": 0.9, "class_id": 0},
        ],
        100,
        80,
    )
    assert left[0] == 10 and right[0] == 70


def test_sampling_keeps_fixed_total_query_count_when_only_one_bbox(tmp_path):
    image = np.full((64, 96, 3), 180, np.uint8)
    image[20:45, 10:25] = 20
    det = YoloDetectionResult(
        job=make_segment(),
        first_frame_rgb=image,
        left_bbox_xyxy=np.array([8, 18, 28, 48], np.float32),
        left_bbox_valid=True,
        left_confidence=0.9,
        right_bbox_xyxy=None,
        right_bbox_valid=False,
        right_confidence=float("nan"),
        raw_detections=[],
        assignment_method="test",
        image_height=64,
        image_width=96,
    )
    result = sample_queries(det, make_cfg(tmp_path))
    assert result.query_xy.shape == (30, 2)
    assert result.left_count + result.right_count + result.env_count == 30
    assert result.left_count <= 10
    assert result.right_count == 0


def test_query_builder_can_fill_missing_environment_points():
    left = np.array([[1, 1], [2, 2]], np.float32)
    right = np.array([[5, 5]], np.float32)
    env = np.array([[1, 1], [3, 3]], np.float32)
    query_xy, query_group, _layout = build_query_set(left, right, env, 8, 10, 10, fill_missing=True, fill_seed=7)
    assert query_xy.shape == (8, 2)
    assert np.count_nonzero(query_group == 0) == 2
    assert np.count_nonzero(query_group == 1) == 1
    assert np.count_nonzero(query_group == 2) == 5


def test_cotracker_output_normalization_accepts_btn_and_bnt_layouts():
    tracks_btn = np.zeros((2, 4, 3, 2), np.float32)
    vis_btn = np.ones((2, 4, 3), bool)
    tracks, vis = normalize_cotracker_output(tracks_btn, vis_btn, 2, 3)
    assert tracks.shape == (2, 3, 4, 2)
    assert vis.shape == (2, 3, 4)
    tracks_bnt = np.zeros((2, 3, 4, 2), np.float32)
    vis_bnt = np.ones((2, 3, 4), bool)
    tracks, vis = normalize_cotracker_output(tracks_bnt, vis_bnt, 2, 3)
    assert tracks.shape == (2, 3, 4, 2)
    assert vis.shape == (2, 3, 4)


def test_shared_array_roundtrip():
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    ref = create_shared_array(array, owner="test", debug_id="roundtrip")
    try:
        assert np.array_equal(copy_shared_array(ref), array)
    finally:
        release_shared_array(ref)


def test_segment_cache_roundtrip_and_validation(tmp_path):
    cfg = make_cfg(tmp_path)
    track = TrackResult(
        job=make_segment(),
        tracks_xy=np.zeros((3, 4, 2), np.float32),
        visibility=np.ones((3, 4), bool),
        confidence=None,
        query_xy=np.array([[1, 1], [2, 2], [3, 3]], np.float32),
        query_group=np.array([0, 1, 2], np.int16),
        detections=None,
        sampling_features={"left": {}, "right": {}, "environment": {}},
        sampling_stats={},
        image_height=10,
        image_width=10,
    )
    seg = filtered_to_segment_result(filter_track_result(track, cfg))
    path = tmp_path / "segment.npz"
    write_segment_cache(path, seg, "abc")
    assert validate_segment_cache(path, seg.job, "abc")
    loaded, fp = read_segment_cache(path)
    assert fp == "abc"
    assert loaded.job.segment_id == seg.job.segment_id
    assert loaded.groups["left"]["tracks_xy_raw"].shape == (1, 4, 2)


def test_cotracker_batches_normal_segments_and_handles_tail(tmp_path):
    cfg = make_cfg(tmp_path)
    runner = PipelineRunner(cfg)
    calls = []

    class FakeCoTracker:
        def track_batch(self, batch):
            calls.append((batch.batch_size, batch.is_tail))
            out = []
            for item in batch.items:
                n = item.query_xy.shape[0]
                t = item.job.frame_count
                out.append(
                    TrackResult(
                        job=item.job,
                        tracks_xy=np.zeros((n, t, 2), np.float32),
                        visibility=np.ones((n, t), bool),
                        confidence=None,
                        query_xy=item.query_xy,
                        query_group=item.query_group,
                        detections=item.detections,
                        sampling_features=item.sampling_features,
                        sampling_stats=item.sampling_stats,
                        image_height=item.image_height,
                        image_width=item.image_width,
                    )
                )
            return out, 0.0, None

    runner.cotracker_model = FakeCoTracker()
    items = []
    for sid, t in [(0, 4), (1, 4), (2, 4), (3, 2)]:
        items.append(
            DecodedTrackItem(
                job=make_segment(sid, t),
                frame_ref=SharedArrayRef("unused", (t, 8, 8, 3), "uint8", t * 8 * 8 * 3, "test", str(sid)),
                query_xy=np.zeros((30, 2), np.float32),
                query_group=np.zeros((30,), np.int16),
                detections=None,
                sampling_features={"left": {}, "right": {}, "environment": {}},
                sampling_stats={},
                image_height=8,
                image_width=8,
            )
        )
    runner._process_cotracker(items)
    assert calls == [(2, False), (1, True), (1, False)]
