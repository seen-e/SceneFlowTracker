from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_flow_tracker.storage.schema import MotionState, TrackState


def counts(arr: np.ndarray, enum_cls) -> dict[str, int]:
    return {member.name: int((arr == int(member)).sum()) for member in enum_cls}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a SceneFlowTracker NPZ.")
    parser.add_argument("npz_path", type=Path)
    args = parser.parse_args()
    with np.load(args.npz_path, allow_pickle=False) as data:
        report = {
            "schema_version": str(data["schema_version"].item()),
            "dataset": str(data["dataset"].item()),
            "episode_id": str(data["episode_id"].item()),
            "view_key": str(data["view_key"].item()),
            "fps": float(data["fps"]),
            "segment_count": int(len(data["segment_ids"])),
            "segment_lengths": data["segment_lengths"].astype(int).tolist(),
            "yolo_bboxes_shape": list(data["yolo_bboxes"].shape),
            "groups": {},
        }
        for group in ("left", "right", "env"):
            report["groups"][group] = {
                "num_points": data[f"{group}_num_points"].astype(int).tolist(),
                "query_xy_shape": list(data[f"{group}_query_xy"].shape),
                "tracks_raw_shape": list(data[f"{group}_tracks_raw"].shape),
                "track_state": counts(data[f"{group}_track_state"], TrackState),
                "motion_state": counts(data[f"{group}_motion_state"], MotionState),
                "usable": int(data[f"{group}_usable"].sum()),
            }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
