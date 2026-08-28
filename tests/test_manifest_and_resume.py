import json

from scene_flow_tracker.config import deep_update, DEFAULT_CONFIG, load_config
from scene_flow_tracker.jobs import EpisodeJob
from scene_flow_tracker.manifest import load_episode_jobs
from scene_flow_tracker.storage.writers import is_episode_done, output_npz_path, output_summary_path


def test_manifest_uses_video_segments_as_authoritative_source(tmp_path):
    video = tmp_path / "file-000.mp4"
    video.write_bytes(b"not used")
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps([{
        "dataset": "abc",
        "episode_id": "abc__episode_000001",
        "episode_index": 1,
        "frame_count": 30,
        "fps": 29.0,
        "video_path": {"observation.images.top": "metadata-only.mp4"},
        "video_segments": {"observation.images.top": {"video_path": str(video), "start_frame": 3329, "end_frame": 3359, "fps": 30.0}},
    }]), encoding="utf-8")
    cfg = deep_update(DEFAULT_CONFIG, {
        "input": {"manifest_path": str(manifest), "strict_validation": True},
        "models": {"yolo": {"model_path": str(video)}, "cotracker": {"model_path": str(video)}},
        "output": {"output_root": str(tmp_path / "out")},
    })
    jobs, invalid = load_episode_jobs(cfg)
    assert not invalid
    assert jobs[0].physical_video_path == str(video)
    assert jobs[0].source_start_frame == 3329
    assert jobs[0].source_end_frame == 3359
    assert jobs[0].manifest_fps == 30.0


def test_yaml_view_key_selects_only_requested_view(tmp_path):
    top = tmp_path / "top.mp4"
    left = tmp_path / "left.mp4"
    model = tmp_path / "model.pt"
    for path in (top, left, model):
        path.write_bytes(b"x")
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps([{
        "dataset": "abc",
        "episode_id": "abc__episode_000002",
        "episode_index": 2,
        "frame_count": 20,
        "fps": 30.0,
        "video_path": {
            "observation.images.top": "metadata-top.mp4",
            "observation.images.left_wrist": "metadata-left.mp4",
        },
        "video_segments": {
            "observation.images.top": {"video_path": str(top), "start_frame": 0, "end_frame": 20, "fps": 30.0},
            "observation.images.left_wrist": {"video_path": str(left), "start_frame": 100, "end_frame": 120, "fps": 30.0},
        },
    }]), encoding="utf-8")
    cfg = deep_update(DEFAULT_CONFIG, {
        "input": {"manifest_path": str(manifest), "view_key": "observation.images.left_wrist"},
        "models": {"yolo": {"model_path": str(model)}, "cotracker": {"model_path": str(model)}},
        "output": {"output_root": str(tmp_path / "out")},
    })
    jobs, _ = load_episode_jobs(cfg)
    assert jobs[0].view_key == "observation.images.left_wrist"
    assert jobs[0].physical_video_path == str(left)
    assert jobs[0].source_start_frame == 100
    assert jobs[0].source_end_frame == 120


def test_resume_detects_completed_episode(tmp_path):
    video = tmp_path / "model.pt"
    video.write_bytes(b"x")

    ep = EpisodeJob("abc", "episode", 0, None, None, None, "observation.images.top", "/tmp/a.mp4", 0, 1, 30, 30, 1)
    out = tmp_path / "out"
    path = output_npz_path(out, ep)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"npz")
    output_summary_path(out, ep).write_text("{}", encoding="utf-8")
    assert is_episode_done(out, ep)


def test_resume_scan_builds_completed_set_without_per_episode_probe(tmp_path):
    from scene_flow_tracker.storage.resume import scan_completed_episode_views

    eps = [
        EpisodeJob("abc", "ep0", 0, None, None, None, "observation.images.top", "/tmp/a.mp4", 0, 1, 30, 30, 1),
        EpisodeJob("abc", "ep1", 1, None, None, None, "observation.images.top", "/tmp/a.mp4", 0, 1, 30, 30, 1),
        EpisodeJob("abc", "ep2", 2, None, None, None, "observation.images.left_wrist", "/tmp/a.mp4", 0, 1, 30, 30, 1),
    ]
    out = tmp_path / "out"
    for ep in eps[:2]:
        path = output_npz_path(out, ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"npz")
    output_summary_path(out, eps[0]).write_text("{}", encoding="utf-8")
    scan = scan_completed_episode_views(out, eps)
    assert scan.is_completed(eps[0])
    assert not scan.is_completed(eps[1])
    assert not scan.is_completed(eps[2])
    assert scan.missing_summary_files == 1


def test_main_config_includes_algorithm_configs(tmp_path):
    manifest = tmp_path / "episodes.json"
    yolo = tmp_path / "yolo.pt"
    cot = tmp_path / "cot.pth"
    for path in (manifest, yolo, cot):
        path.write_text("[]" if path == manifest else "x", encoding="utf-8")
    (tmp_path / "sampling.yaml").write_text(
        "robot_sampling:\n  left_robot:\n    target_points: 7\nenvironment_sampling:\n  target_points: 9\n",
        encoding="utf-8",
    )
    (tmp_path / "trajectory_filter.yaml").write_text(
        "trajectory_filter:\n  motion_classifier:\n    version: v2\n  output_policy:\n    keep_jitter: false\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
input:
  manifest_path: {manifest}
models:
  yolo:
    model_path: {yolo}
  cotracker:
    model_path: {cot}
algorithm_configs:
  sampling: sampling.yaml
  trajectory_filter: trajectory_filter.yaml
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg["robot_sampling"]["left_robot"]["target_points"] == 7
    assert cfg["environment_sampling"]["target_points"] == 9
    assert cfg["trajectory_filter"]["motion_classifier"]["version"] == "v2"
