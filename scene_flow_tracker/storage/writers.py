from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..jobs import EpisodeJob, SegmentResult
from .schema import (
    CANDIDATE_LEVEL_MAP,
    ENUM_METADATA,
    ENV_SAMPLING_FEATURES,
    FILTER_FEATURE_ALIASES,
    FRAME_RANGE_SEMANTICS,
    GROUP_TO_RESULT_KEY,
    GROUPS,
    MOTION_STATE_MAP,
    ROBOT_SAMPLING_FEATURES,
    SCHEMA_VERSION,
    TRACK_COORDINATE_ORDER,
    TRACK_COORDINATE_SPACE,
    TRACK_STATE_MAP,
    YOLO_CLASS_NAMES,
    GroupStatus,
    MotionState,
    SegmentStatus,
    TrackState,
)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def episode_output_dir(output_root: Path, episode: EpisodeJob) -> Path:
    return output_root / episode.dataset / episode.episode_id


def safe_view_name(view_key: str) -> str:
    return view_key.replace("/", "_")


def output_npz_path(output_root: Path, episode: EpisodeJob) -> Path:
    return episode_output_dir(output_root, episode) / f"{safe_view_name(episode.view_key)}_scene_tracks.npz"


def output_summary_path(output_root: Path, episode: EpisodeJob) -> Path:
    return episode_output_dir(output_root, episode) / f"{safe_view_name(episode.view_key)}_summary.json"


def is_episode_done(output_root: Path, episode: EpisodeJob) -> bool:
    path = output_npz_path(output_root, episode)
    return path.exists() and path.stat().st_size > 0 and output_summary_path(output_root, episode).exists()


def _string_array(value: Any) -> np.ndarray:
    return np.array("" if value is None else str(value))


def _map_string_states(values: Any, mapping: dict[str, Any], dtype=np.int8) -> np.ndarray:
    arr = np.asarray(values)
    out = np.zeros(arr.shape, dtype=dtype)
    for idx, value in np.ndenumerate(arr.astype(str)):
        enum_value = mapping.get(value.strip().lower(), 0)
        out[idx] = int(enum_value)
    return out


def _as_points(group: dict[str, Any] | None) -> np.ndarray:
    if not group:
        return np.empty((0, 2), np.float32)
    query = np.asarray(group.get("query_xy", []), dtype=np.float32)
    if query.size == 0:
        return np.empty((0, 2), np.float32)
    return query.reshape(-1, 2)


def _as_tracks(group: dict[str, Any] | None, key: str, n: int, t: int) -> np.ndarray:
    if not group or key not in group:
        return np.full((n, t, 2), np.nan, np.float32)
    tracks = np.asarray(group[key], dtype=np.float32)
    if tracks.size == 0:
        return np.empty((0, t, 2), np.float32)
    if tracks.shape != (n, t, 2):
        raise ValueError(f"{key} expects [N,T,2], got {tracks.shape}")
    return tracks


def _as_visibility(group: dict[str, Any] | None, n: int, t: int) -> np.ndarray:
    if not group or "visibility" not in group:
        return np.empty((n, t), bool)
    visibility = np.asarray(group["visibility"], dtype=bool)
    if visibility.size == 0:
        return np.empty((0, t), bool)
    if visibility.shape != (n, t):
        raise ValueError(f"visibility expects [N,T], got {visibility.shape}")
    return visibility


def _optional_confidence(group: dict[str, Any] | None, n: int, t: int) -> np.ndarray | None:
    if not group or "cotracker_confidence" not in group:
        return None
    conf = np.asarray(group["cotracker_confidence"], dtype=np.float32)
    if conf.size == 0:
        return np.empty((0, t), np.float32)
    if conf.shape != (n, t):
        raise ValueError(f"cotracker_confidence expects [N,T], got {conf.shape}")
    return conf


def _state_array(group: dict[str, Any] | None, key: str, n: int, mapping: dict[str, Any]) -> np.ndarray:
    if not group or key not in group:
        return np.zeros((n,), dtype=np.int8)
    values = np.asarray(group[key])
    if values.size == 0:
        return np.zeros((0,), dtype=np.int8)
    if values.shape != (n,):
        raise ValueError(f"{key} expects [N], got {values.shape}")
    return _map_string_states(values, mapping)


def _usable_array(group: dict[str, Any] | None, n: int) -> np.ndarray:
    if not group or "usable_for_robot_scene_flow" not in group:
        return np.zeros((n,), dtype=bool)
    values = np.asarray(group["usable_for_robot_scene_flow"], dtype=bool)
    if values.shape != (n,):
        raise ValueError(f"usable_for_robot_scene_flow expects [N], got {values.shape}")
    return values


def _group_status(res: SegmentResult, name: str, n: int) -> int:
    if res.status != "DONE":
        return int(GroupStatus.PROCESSING_FAILED)
    if name in ("left", "right") and not res.sampling.get(f"{name}_bbox"):
        return int(GroupStatus.NO_DETECTION)
    if n == 0:
        return int(GroupStatus.NO_CANDIDATES)
    return int(GroupStatus.OK)


def _segment_status(res: SegmentResult) -> int:
    if res.status == "DONE":
        return int(SegmentStatus.OK)
    if res.status == "PARTIAL":
        return int(SegmentStatus.PARTIAL)
    return int(SegmentStatus.FAILED)


def _feature_vector(group: dict[str, Any] | None, names: tuple[str, ...], n: int) -> np.ndarray | None:
    if not group:
        return None
    for name in names:
        if name in group:
            values = np.asarray(group[name], dtype=np.float32)
            return values if values.shape == (n,) else None
    features = group.get("motion_features")
    if isinstance(features, list):
        values = np.full((n,), np.nan, dtype=np.float32)
        any_value = False
        for i, item in enumerate(features[:n]):
            if isinstance(item, dict):
                for name in names:
                    if name in item:
                        values[i] = float(item[name])
                        any_value = True
                        break
        return values if any_value else None
    if isinstance(features, dict):
        for name in names:
            if name in features:
                values = np.asarray(features[name], dtype=np.float32)
                return values if values.shape == (n,) else None
    return None


def _sampling_vector(res: SegmentResult, group_name: str, field: str, n: int) -> np.ndarray | None:
    key = GROUP_TO_RESULT_KEY[group_name]
    group_sampling = res.sampling.get(key, {})
    if isinstance(group_sampling, dict) and field in group_sampling:
        values = np.asarray(group_sampling[field], dtype=np.float32)
        return values if values.shape == (n,) else None
    result_group = res.groups.get(key)
    if result_group and field in result_group:
        values = np.asarray(result_group[field], dtype=np.float32)
        return values if values.shape == (n,) else None
    return None


def _candidate_level_vector(res: SegmentResult, group_name: str, n: int) -> np.ndarray | None:
    raw = None
    key = GROUP_TO_RESULT_KEY[group_name]
    group_sampling = res.sampling.get(key, {})
    if isinstance(group_sampling, dict):
        raw = group_sampling.get("candidate_level")
    if raw is None and key in res.groups:
        raw = res.groups[key].get("candidate_level")
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.shape != (n,):
        return None
    return _map_string_states(arr, CANDIDATE_LEVEL_MAP, dtype=np.int8)


def _file_short_hash(path: Any) -> str:
    if not path:
        return ""
    p = Path(str(path))
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _config_hash(episode: EpisodeJob) -> str:
    if not episode.raw_record:
        return ""
    blob = json.dumps(json_safe(episode.raw_record), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _collect_image_size(results: list[SegmentResult]) -> tuple[int, int]:
    for res in results:
        width = res.sampling.get("image_width")
        height = res.sampling.get("image_height")
        if width is not None and height is not None:
            return int(width), int(height)
    return -1, -1


def _validate_segment_continuity(episode: EpisodeJob, results: list[SegmentResult]) -> None:
    if not results:
        return
    for prev, cur in zip(results, results[1:]):
        if prev.job.episode_end_frame != cur.job.episode_start_frame:
            raise ValueError(f"episode segment gap/overlap: {prev.job.segment_id}->{cur.job.segment_id}")
        if prev.job.source_end_frame != cur.job.source_start_frame:
            raise ValueError(f"source segment gap/overlap: {prev.job.segment_id}->{cur.job.segment_id}")
    for res in results:
        if res.job.source_start_frame != episode.source_start_frame + res.job.episode_start_frame:
            raise ValueError(f"segment {res.job.segment_id} source/episode offset mismatch")
        if res.job.frame_count != res.job.episode_end_frame - res.job.episode_start_frame:
            raise ValueError(f"segment {res.job.segment_id} episode length mismatch")
        if res.job.frame_count != res.job.source_end_frame - res.job.source_start_frame:
            raise ValueError(f"segment {res.job.segment_id} source length mismatch")


def _put_group(payload: dict[str, Any], results: list[SegmentResult], group_name: str, tmax: int, save_filter_features: bool, save_sampling_features: bool, save_conf: bool) -> dict[str, int]:
    result_key = GROUP_TO_RESULT_KEY[group_name]
    counts = np.asarray([len(_as_points(res.groups.get(result_key))) for res in results], dtype=np.int32)
    nmax = int(counts.max(initial=0))
    s_count = len(results)
    payload[f"{group_name}_num_points"] = counts
    payload[f"{group_name}_point_valid"] = np.zeros((s_count, nmax), dtype=bool)
    payload[f"{group_name}_query_xy"] = np.full((s_count, nmax, 2), np.nan, dtype=np.float32)
    payload[f"{group_name}_tracks_raw"] = np.full((s_count, tmax, nmax, 2), np.nan, dtype=np.float32)
    payload[f"{group_name}_tracks_smooth"] = np.full((s_count, tmax, nmax, 2), np.nan, dtype=np.float32)
    payload[f"{group_name}_visibility"] = np.zeros((s_count, tmax, nmax), dtype=bool)
    payload[f"{group_name}_track_state"] = np.zeros((s_count, nmax), dtype=np.int8)
    payload[f"{group_name}_motion_state"] = np.zeros((s_count, nmax), dtype=np.int8)
    payload[f"{group_name}_usable"] = np.zeros((s_count, nmax), dtype=bool)
    payload[f"{group_name}_group_status"] = np.zeros((s_count,), dtype=np.int8)
    conf_values: list[tuple[int, np.ndarray]] = []

    for s, res in enumerate(results):
        group = res.groups.get(result_key)
        n = int(counts[s])
        t = int(res.job.frame_count)
        payload[f"{group_name}_group_status"][s] = _group_status(res, result_key, n)
        if n == 0:
            continue
        query = _as_points(group)
        raw = _as_tracks(group, "tracks_xy_raw", n, t)
        smooth = _as_tracks(group, "tracks_xy_smooth", n, t)
        visibility = _as_visibility(group, n, t)
        payload[f"{group_name}_point_valid"][s, :n] = True
        payload[f"{group_name}_query_xy"][s, :n] = query
        payload[f"{group_name}_tracks_raw"][s, :t, :n] = np.transpose(raw, (1, 0, 2))
        payload[f"{group_name}_tracks_smooth"][s, :t, :n] = np.transpose(smooth, (1, 0, 2))
        payload[f"{group_name}_visibility"][s, :t, :n] = np.transpose(visibility, (1, 0))
        payload[f"{group_name}_track_state"][s, :n] = _state_array(group, "track_state", n, TRACK_STATE_MAP)
        payload[f"{group_name}_motion_state"][s, :n] = _state_array(group, "motion_state", n, MOTION_STATE_MAP)
        payload[f"{group_name}_usable"][s, :n] = _usable_array(group, n)
        conf = _optional_confidence(group, n, t)
        if conf is not None:
            conf_values.append((s, conf))

    if save_conf and conf_values:
        payload[f"{group_name}_cotracker_confidence"] = np.full((s_count, tmax, nmax), np.nan, dtype=np.float32)
        for s, conf in conf_values:
            t, n = conf.shape[1], conf.shape[0]
            payload[f"{group_name}_cotracker_confidence"][s, :t, :n] = np.transpose(conf, (1, 0))

    if save_sampling_features:
        fields = ENV_SAMPLING_FEATURES if group_name == "env" else ROBOT_SAMPLING_FEATURES
        for field in fields:
            arr = np.full((s_count, nmax), np.nan, dtype=np.float32)
            wrote = False
            for s, res in enumerate(results):
                n = int(counts[s])
                values = _sampling_vector(res, group_name, field, n)
                if values is not None:
                    arr[s, :n] = values
                    wrote = True
            if wrote:
                payload[f"{group_name}_{field}"] = arr
        if group_name != "env":
            arr = np.zeros((s_count, nmax), dtype=np.int8)
            wrote = False
            for s, res in enumerate(results):
                n = int(counts[s])
                values = _candidate_level_vector(res, group_name, n)
                if values is not None:
                    arr[s, :n] = values
                    wrote = True
            if wrote:
                payload[f"{group_name}_candidate_level"] = arr

    if save_filter_features:
        for out_name, aliases in FILTER_FEATURE_ALIASES.items():
            arr = np.full((s_count, nmax), np.nan, dtype=np.float32)
            wrote = False
            for s, res in enumerate(results):
                group = res.groups.get(result_key)
                n = int(counts[s])
                values = _feature_vector(group, aliases, n)
                if values is not None:
                    arr[s, :n] = values
                    wrote = True
            if wrote:
                payload[f"{group_name}_{out_name}"] = arr

    return {
        "total_queries": int(counts.sum()),
        "usable": int(payload[f"{group_name}_usable"].sum()),
        "track_valid": int((payload[f"{group_name}_track_state"] == int(TrackState.VALID)).sum()),
        "track_partial": int((payload[f"{group_name}_track_state"] == int(TrackState.PARTIAL)).sum()),
        "track_failed": int((payload[f"{group_name}_track_state"] == int(TrackState.FAILED)).sum()),
        "static": int((payload[f"{group_name}_motion_state"] == int(MotionState.STATIC)).sum()),
        "moving": int((payload[f"{group_name}_motion_state"] == int(MotionState.MOVING)).sum()),
        "jitter": int((payload[f"{group_name}_motion_state"] == int(MotionState.JITTER)).sum()),
        "uncertain": int((payload[f"{group_name}_motion_state"] == int(MotionState.UNCERTAIN)).sum()),
    }


def _put_yolo(payload: dict[str, Any], results: list[SegmentResult]) -> None:
    s_count, k, c = len(results), 1, len(YOLO_CLASS_NAMES)
    payload["yolo_class_ids"] = np.arange(c, dtype=np.int16)
    payload["yolo_class_names"] = np.asarray(YOLO_CLASS_NAMES)
    payload["yolo_frame_indices"] = np.full((s_count, k), -1, dtype=np.int64)
    payload["yolo_episode_frame_indices"] = np.full((s_count, k), -1, dtype=np.int64)
    payload["yolo_frame_valid"] = np.zeros((s_count, k), dtype=bool)
    payload["yolo_bboxes"] = np.full((s_count, k, c, 4), np.nan, dtype=np.float32)
    payload["yolo_bbox_confidence"] = np.full((s_count, k, c), np.nan, dtype=np.float32)
    payload["yolo_bbox_valid"] = np.zeros((s_count, k, c), dtype=bool)
    for s, res in enumerate(results):
        if res.status != "DONE":
            continue
        payload["yolo_frame_indices"][s, 0] = res.job.source_start_frame
        payload["yolo_episode_frame_indices"][s, 0] = res.job.episode_start_frame
        payload["yolo_frame_valid"][s, 0] = True
        for det in res.detections:
            slot = det.get("slot")
            if slot not in ("left", "right"):
                continue
            cls = 0 if slot == "left" else 1
            payload["yolo_bboxes"][s, 0, cls] = np.asarray(det.get("bbox_xyxy", [np.nan] * 4), dtype=np.float32)
            payload["yolo_bbox_confidence"][s, 0, cls] = float(det.get("confidence", np.nan))
            payload["yolo_bbox_valid"][s, 0, cls] = bool(np.all(np.isfinite(payload["yolo_bboxes"][s, 0, cls])))


def build_episode_payload(episode: EpisodeJob, results: list[SegmentResult], cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    results = sorted(results, key=lambda r: r.job.segment_id)
    _validate_segment_continuity(episode, results)
    s_count = len(results)
    tmax = max((r.job.frame_count for r in results), default=0)
    image_width, image_height = _collect_image_size(results)
    output_cfg = (cfg or {}).get("output", {}) if cfg else {}
    save_filter_features = bool(output_cfg.get("save_filter_features", output_cfg.get("save_features", True)))
    save_sampling_features = bool(output_cfg.get("save_sampling_features", True))
    save_conf = bool(output_cfg.get("save_cotracker_confidence", True))

    payload: dict[str, Any] = {
        "schema_version": _string_array(SCHEMA_VERSION),
        "dataset": _string_array(episode.dataset),
        "episode_id": _string_array(episode.episode_id),
        "episode_index": np.int64(episode.episode_index),
        "task_index": np.int64(-1 if episode.task_index is None else episode.task_index),
        "view_key": _string_array(episode.view_key),
        "fps": np.float32(episode.effective_fps),
        "manifest_fps": np.float32(episode.manifest_fps),
        "image_width": np.int32(image_width),
        "image_height": np.int32(image_height),
        "source_video_path": _string_array(episode.physical_video_path),
        "episode_source_start_frame": np.int64(episode.source_start_frame),
        "episode_source_end_frame": np.int64(episode.source_end_frame),
        "frame_range_semantics": _string_array(FRAME_RANGE_SEMANTICS),
        "track_coordinate_order": _string_array(TRACK_COORDINATE_ORDER),
        "track_coordinate_space": _string_array(TRACK_COORDINATE_SPACE),
        "segment_ids": np.asarray([r.job.segment_id for r in results], dtype=np.int32),
        "episode_start_frames": np.asarray([r.job.episode_start_frame for r in results], dtype=np.int64),
        "episode_end_frames": np.asarray([r.job.episode_end_frame for r in results], dtype=np.int64),
        "source_start_frames": np.asarray([r.job.source_start_frame for r in results], dtype=np.int64),
        "source_end_frames": np.asarray([r.job.source_end_frame for r in results], dtype=np.int64),
        "segment_lengths": np.asarray([r.job.frame_count for r in results], dtype=np.int32),
        "segment_status": np.asarray([_segment_status(r) for r in results], dtype=np.int8),
        "schema_metadata_json": _string_array(json.dumps(ENUM_METADATA, ensure_ascii=False, sort_keys=True)),
    }
    payload["episode_frame_indices"] = np.full((s_count, tmax), -1, dtype=np.int64)
    payload["source_frame_indices"] = np.full((s_count, tmax), -1, dtype=np.int64)
    for s, res in enumerate(results):
        t = res.job.frame_count
        payload["episode_frame_indices"][s, :t] = np.arange(res.job.episode_start_frame, res.job.episode_end_frame, dtype=np.int64)
        payload["source_frame_indices"][s, :t] = np.arange(res.job.source_start_frame, res.job.source_end_frame, dtype=np.int64)

    _put_yolo(payload, results)
    group_stats = {
        "left": _put_group(payload, results, "left", tmax, save_filter_features, save_sampling_features, save_conf),
        "right": _put_group(payload, results, "right", tmax, save_filter_features, save_sampling_features, save_conf),
        "environment": _put_group(payload, results, "env", tmax, save_filter_features, save_sampling_features, save_conf),
    }

    failed_segments = [
        {"segment_id": r.job.segment_id, "error_code": r.error_code, "message": r.error_message}
        for r in results
        if r.status != "DONE"
    ]
    segment_status = payload["segment_status"]
    provenance = {
        "pipeline_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "config_hash": _config_hash(episode),
        "yolo_checkpoint": "",
        "yolo_checkpoint_hash": "",
        "cotracker_checkpoint": "",
        "cotracker_checkpoint_hash": "",
    }
    models = (cfg or {}).get("models", {}) if cfg else {}
    yolo_path = models.get("yolo", {}).get("model_path") if isinstance(models.get("yolo", {}), dict) else None
    cotracker_path = models.get("cotracker", {}).get("model_path") if isinstance(models.get("cotracker", {}), dict) else None
    if yolo_path:
        provenance["yolo_checkpoint"] = Path(str(yolo_path)).name
        provenance["yolo_checkpoint_hash"] = _file_short_hash(yolo_path)
    if cotracker_path:
        provenance["cotracker_checkpoint"] = Path(str(cotracker_path)).name
        provenance["cotracker_checkpoint_hash"] = _file_short_hash(cotracker_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": episode.dataset,
        "episode_id": episode.episode_id,
        "episode_index": episode.episode_index,
        "task_index": episode.task_index,
        "task": episode.task,
        "instruction": episode.instruction,
        "view_key": episode.view_key,
        "source_video_path": episode.physical_video_path,
        "episode_source_start_frame": episode.source_start_frame,
        "episode_source_end_frame": episode.source_end_frame,
        "fps": episode.effective_fps,
        "manifest_fps": episode.manifest_fps,
        "image_width": image_width,
        "image_height": image_height,
        "segment_count": s_count,
        "frame_range_semantics": FRAME_RANGE_SEMANTICS,
        "segments_ok": int((segment_status == int(SegmentStatus.OK)).sum()),
        "segments_partial": int((segment_status == int(SegmentStatus.PARTIAL)).sum()),
        "segments_failed": int((segment_status == int(SegmentStatus.FAILED)).sum()),
        "failed_segment_ids": [item["segment_id"] for item in failed_segments],
        "failed_segments": failed_segments,
        "left": group_stats["left"],
        "right": group_stats["right"],
        "environment": group_stats["environment"],
        "schema": {
            "enums": ENUM_METADATA,
            "diagnostic_only_features": ["path_efficiency", "direction_reversal_ratio"],
            "shapes": {
                "tracks": "[S,Tmax,Nmax,2]",
                "visibility": "[S,Tmax,Nmax]",
                "query_xy": "[S,Nmax,2]",
            },
        },
        "processing": provenance,
        "segments": [segment_summary(r) for r in results],
    }
    payload["summary_json"] = _string_array(json.dumps(json_safe(summary), ensure_ascii=False, sort_keys=True))
    return payload, summary


def validate_payload(payload: dict[str, Any], episode: EpisodeJob) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version",
        "segment_ids",
        "episode_start_frames",
        "episode_end_frames",
        "source_start_frames",
        "source_end_frames",
        "segment_lengths",
        "episode_frame_indices",
        "source_frame_indices",
        "segment_status",
        "yolo_bboxes",
    ]
    for key in required:
        if key not in payload:
            errors.append(f"missing key: {key}")
    if errors:
        return errors
    starts = payload["episode_start_frames"]
    ends = payload["episode_end_frames"]
    source_starts = payload["source_start_frames"]
    source_ends = payload["source_end_frames"]
    lengths = payload["segment_lengths"]
    if not np.all(ends - starts == lengths):
        errors.append("episode segment length mismatch")
    if not np.all(source_ends - source_starts == lengths):
        errors.append("source segment length mismatch")
    if len(starts) and not np.all(source_starts == int(episode.source_start_frame) + starts):
        errors.append("source frame offset mismatch")
    if len(starts) > 1:
        if not np.all(ends[:-1] == starts[1:]):
            errors.append("episode segments have gap or overlap")
        if not np.all(source_ends[:-1] == source_starts[1:]):
            errors.append("source segments have gap or overlap")
    for group_name in GROUPS:
        nkey = f"{group_name}_num_points"
        vkey = f"{group_name}_point_valid"
        if nkey not in payload or vkey not in payload:
            errors.append(f"missing group keys: {group_name}")
            continue
        if not np.all(payload[vkey].sum(axis=1) == payload[nkey]):
            errors.append(f"{group_name} point_valid count mismatch")
        invalid = ~payload[vkey]
        q = payload[f"{group_name}_query_xy"]
        if invalid.size and not np.all(np.isnan(q[invalid])):
            errors.append(f"{group_name} query padding is not NaN")
        tracks = payload[f"{group_name}_tracks_raw"]
        for s, length in enumerate(lengths):
            if length < tracks.shape[1] and not np.all(np.isnan(tracks[s, int(length) :, :, :])):
                errors.append(f"{group_name} frame padding is not NaN")
                break
    return errors


def write_episode_outputs(output_root: Path, episode: EpisodeJob, results: list[SegmentResult], atomic: bool = True, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    out_dir = episode_output_dir(output_root, episode)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, summary = build_episode_payload(episode, results, cfg)
    errors = validate_payload(payload, episode)
    if errors:
        raise ValueError("NPZ payload validation failed: " + "; ".join(errors))
    npz_path = output_npz_path(output_root, episode)
    tmp = npz_path.with_suffix(".npz.tmp")
    compression = ((cfg or {}).get("output", {}) if cfg else {}).get("compression", "compressed")
    with tmp.open("wb") as f:
        if compression == "stored":
            np.savez(f, **payload)
        else:
            np.savez_compressed(f, **payload)
    if atomic:
        os.replace(tmp, npz_path)
    else:
        shutil.move(tmp, npz_path)
    summary["output_npz"] = str(npz_path)
    summary_path = output_summary_path(output_root, episode)
    tmp_summary = summary_path.with_suffix(".json.tmp")
    tmp_summary.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if atomic:
        os.replace(tmp_summary, summary_path)
    else:
        shutil.move(tmp_summary, summary_path)
    return summary


def segment_summary(res: SegmentResult) -> dict[str, Any]:
    item: dict[str, Any] = {
        "segment_id": res.job.segment_id,
        "status": res.status,
        "episode_start_frame": res.job.episode_start_frame,
        "episode_end_frame": res.job.episode_end_frame,
        "source_start_frame": res.job.source_start_frame,
        "source_end_frame": res.job.source_end_frame,
        "frame_count": res.job.frame_count,
        "error_code": res.error_code,
        "error_message": res.error_message,
        "timings": res.timings,
        "sampling": res.sampling,
    }
    for side, result_key in GROUP_TO_RESULT_KEY.items():
        group = res.groups.get(result_key)
        if not group:
            continue
        item["environment" if side == "env" else side] = {
            "num_points": int(len(_as_points(group))),
            "track_state": _string_counts(group.get("track_state", [])),
            "motion_state": _string_counts(group.get("motion_state", [])),
            "usable": int(np.asarray(group.get("usable_for_robot_scene_flow", []), dtype=bool).sum()),
        }
    return item


def _string_counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in np.asarray(values).astype(str).tolist():
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def append_processing_manifest(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(item), ensure_ascii=False) + "\n")


def append_processing_manifest_many(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(json_safe(item), ensure_ascii=False) + "\n")
