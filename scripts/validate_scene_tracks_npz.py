from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_flow_tracker.storage.schema import SCHEMA_VERSION


REQUIRED_KEYS = (
    "schema_version",
    "dataset",
    "episode_id",
    "view_key",
    "fps",
    "source_video_path",
    "episode_source_start_frame",
    "episode_source_end_frame",
    "frame_range_semantics",
    "segment_ids",
    "episode_start_frames",
    "episode_end_frames",
    "source_start_frames",
    "source_end_frames",
    "segment_lengths",
    "episode_frame_indices",
    "source_frame_indices",
    "segment_status",
    "yolo_frame_indices",
    "yolo_frame_valid",
    "yolo_bboxes",
    "yolo_bbox_confidence",
    "yolo_bbox_valid",
)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        for key in REQUIRED_KEYS:
            if key not in data:
                errors.append(f"missing key: {key}")
        if errors:
            return errors
        if str(data["schema_version"].item()) != SCHEMA_VERSION:
            errors.append(f"schema_version is not {SCHEMA_VERSION}")
        starts = data["episode_start_frames"]
        ends = data["episode_end_frames"]
        source_starts = data["source_start_frames"]
        source_ends = data["source_end_frames"]
        lengths = data["segment_lengths"]
        if not np.all(ends - starts == lengths):
            errors.append("episode frame ranges do not match segment_lengths")
        if not np.all(source_ends - source_starts == lengths):
            errors.append("source frame ranges do not match segment_lengths")
        if len(starts) > 1 and not np.all(ends[:-1] == starts[1:]):
            errors.append("episode segments contain gap or overlap")
        if len(starts) > 1 and not np.all(source_ends[:-1] == source_starts[1:]):
            errors.append("source segments contain gap or overlap")
        episode_source_start = int(data["episode_source_start_frame"])
        if not np.all(source_starts == episode_source_start + starts):
            errors.append("source_start_frames are not episode_source_start_frame + episode_start_frames")
        for s, length in enumerate(lengths.astype(int)):
            if not np.all(data["episode_frame_indices"][s, :length] == np.arange(starts[s], ends[s])):
                errors.append(f"bad episode_frame_indices at segment {s}")
            if not np.all(data["source_frame_indices"][s, :length] == np.arange(source_starts[s], source_ends[s])):
                errors.append(f"bad source_frame_indices at segment {s}")
            if length < data["episode_frame_indices"].shape[1]:
                if not np.all(data["episode_frame_indices"][s, length:] == -1):
                    errors.append(f"episode frame padding is not -1 at segment {s}")
                if not np.all(data["source_frame_indices"][s, length:] == -1):
                    errors.append(f"source frame padding is not -1 at segment {s}")
        for group in ("left", "right", "env"):
            for suffix in ("num_points", "point_valid", "query_xy", "tracks_raw", "tracks_smooth", "visibility", "track_state", "motion_state", "usable", "group_status"):
                key = f"{group}_{suffix}"
                if key not in data:
                    errors.append(f"missing key: {key}")
            if errors:
                continue
            npoints = data[f"{group}_num_points"]
            valid = data[f"{group}_point_valid"]
            if not np.all(valid.sum(axis=1) == npoints):
                errors.append(f"{group}: point_valid count mismatch")
            if data[f"{group}_tracks_raw"].shape[:2] != (len(starts), data["episode_frame_indices"].shape[1]):
                errors.append(f"{group}: bad tracks_raw [S,T] shape")
            invalid = ~valid
            if invalid.size and not np.all(np.isnan(data[f"{group}_query_xy"][invalid])):
                errors.append(f"{group}: query padding is not NaN")
            tracks_raw = data[f"{group}_tracks_raw"]
            tracks_smooth = data[f"{group}_tracks_smooth"]
            for s, length in enumerate(lengths.astype(int)):
                if length < tracks_raw.shape[1]:
                    if not np.all(np.isnan(tracks_raw[s, length:, :, :])):
                        errors.append(f"{group}: tracks_raw frame padding is not NaN at segment {s}")
                    if not np.all(np.isnan(tracks_smooth[s, length:, :, :])):
                        errors.append(f"{group}: tracks_smooth frame padding is not NaN at segment {s}")
                    if np.any(data[f"{group}_visibility"][s, length:, :]):
                        errors.append(f"{group}: visibility frame padding is not False at segment {s}")
            for state_key, max_value in ((f"{group}_track_state", 3), (f"{group}_motion_state", 4), (f"{group}_group_status", 5)):
                arr = data[state_key]
                if np.any((arr < 0) | (arr > max_value)):
                    errors.append(f"{state_key}: enum value out of range")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a SceneFlowTracker NPZ.")
    parser.add_argument("npz_path", type=Path)
    args = parser.parse_args()
    errors = validate(args.npz_path)
    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
