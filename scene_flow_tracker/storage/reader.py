from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .schema import GROUP_TO_RESULT_KEY, GROUPS, SCHEMA_VERSION


class EpisodeSceneTrackReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)
        self.schema_version = str(self.data["schema_version"].item()) if "schema_version" in self.data else "legacy"
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported scene track schema: {self.schema_version}")
        self.metadata = self._read_metadata()

    def close(self) -> None:
        self.data.close()

    def __enter__(self) -> "EpisodeSceneTrackReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def num_segments(self) -> int:
        return int(len(self.data["segment_ids"]))

    def _scalar(self, key: str) -> Any:
        value = self.data[key]
        if value.shape == ():
            return value.item()
        return value

    def _read_metadata(self) -> dict[str, Any]:
        keys = [
            "schema_version",
            "dataset",
            "episode_id",
            "episode_index",
            "task_index",
            "view_key",
            "fps",
            "manifest_fps",
            "image_width",
            "image_height",
            "source_video_path",
            "episode_source_start_frame",
            "episode_source_end_frame",
            "frame_range_semantics",
            "track_coordinate_order",
            "track_coordinate_space",
        ]
        return {key: self._scalar(key) for key in keys if key in self.data}

    def _index_for_segment(self, segment_id: int) -> int:
        matches = np.flatnonzero(self.data["segment_ids"] == int(segment_id))
        if len(matches) != 1:
            raise KeyError(f"segment_id not found: {segment_id}")
        return int(matches[0])

    def get_segment(self, segment_id: int) -> dict[str, Any]:
        s = self._index_for_segment(segment_id)
        length = int(self.data["segment_lengths"][s])
        return {
            "segment_id": int(self.data["segment_ids"][s]),
            "episode_start_frame": int(self.data["episode_start_frames"][s]),
            "episode_end_frame": int(self.data["episode_end_frames"][s]),
            "source_start_frame": int(self.data["source_start_frames"][s]),
            "source_end_frame": int(self.data["source_end_frames"][s]),
            "segment_length": length,
            "episode_frame_indices": self.data["episode_frame_indices"][s, :length].copy(),
            "source_frame_indices": self.data["source_frame_indices"][s, :length].copy(),
            "segment_status": int(self.data["segment_status"][s]),
        }

    def _get_group(self, segment_id: int, group_name: str) -> dict[str, Any]:
        if group_name not in GROUPS:
            raise ValueError(f"unknown group: {group_name}")
        s = self._index_for_segment(segment_id)
        n = int(self.data[f"{group_name}_num_points"][s])
        t = int(self.data["segment_lengths"][s])
        out = {
            "query_xy": self.data[f"{group_name}_query_xy"][s, :n].copy(),
            "tracks_raw": self.data[f"{group_name}_tracks_raw"][s, :t, :n].copy(),
            "tracks_smooth": self.data[f"{group_name}_tracks_smooth"][s, :t, :n].copy(),
            "visibility": self.data[f"{group_name}_visibility"][s, :t, :n].copy(),
            "track_state": self.data[f"{group_name}_track_state"][s, :n].copy(),
            "motion_state": self.data[f"{group_name}_motion_state"][s, :n].copy(),
            "usable": self.data[f"{group_name}_usable"][s, :n].copy(),
            "group_status": int(self.data[f"{group_name}_group_status"][s]),
        }
        conf_key = f"{group_name}_cotracker_confidence"
        if conf_key in self.data:
            out["cotracker_confidence"] = self.data[conf_key][s, :t, :n].copy()
        for key in self.data.files:
            prefix = f"{group_name}_"
            if key.startswith(prefix) and key not in {
                f"{group_name}_num_points",
                f"{group_name}_point_valid",
                f"{group_name}_query_xy",
                f"{group_name}_tracks_raw",
                f"{group_name}_tracks_smooth",
                f"{group_name}_visibility",
                f"{group_name}_track_state",
                f"{group_name}_motion_state",
                f"{group_name}_usable",
                f"{group_name}_group_status",
                conf_key,
            }:
                arr = self.data[key]
                if arr.ndim == 2 and arr.shape[0] == self.num_segments:
                    out[key[len(prefix) :]] = arr[s, :n].copy()
        return out

    def get_left(self, segment_id: int) -> dict[str, Any]:
        return self._get_group(segment_id, "left")

    def get_right(self, segment_id: int) -> dict[str, Any]:
        return self._get_group(segment_id, "right")

    def get_environment(self, segment_id: int) -> dict[str, Any]:
        return self._get_group(segment_id, "env")
