from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_flow_tracker.video_decode import decode_frames_rgb


GROUP_COLORS_BGR = {
    "left": (30, 60, 255),
    "right": (255, 90, 20),
    "env": (60, 210, 90),
}

NON_USABLE_COLOR_BGR = (150, 150, 150)


def _scalar_str(value: np.ndarray) -> str:
    return str(value.item()) if hasattr(value, "shape") and value.shape == () else str(value)


def _infer_summary_path(npz_path: Path) -> Path:
    name = npz_path.name
    if name.endswith("_scene_tracks.npz"):
        return npz_path.with_name(name[: -len("_scene_tracks.npz")] + "_summary.json")
    return npz_path.with_suffix(".json")


def _parse_groups(value: str) -> list[str]:
    groups = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"left", "right", "env"}
    bad = [item for item in groups if item not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown groups: {bad}; choose from left,right,env")
    return groups


def _track_key(data: np.lib.npyio.NpzFile, group: str, source: str) -> str:
    preferred = f"{group}_tracks_{source}"
    if preferred in data.files:
        return preferred
    fallback = f"{group}_tracks_raw"
    if fallback in data.files:
        return fallback
    raise KeyError(f"missing tracks for group={group}, source={source}")


def _select_indices(data: np.lib.npyio.NpzFile, group: str, segment_idx: int, max_points: int, draw_non_usable: bool) -> np.ndarray:
    n = int(data[f"{group}_num_points"][segment_idx])
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    valid = data[f"{group}_point_valid"][segment_idx, :n].astype(bool)
    if draw_non_usable:
        mask = valid
    else:
        mask = valid & data[f"{group}_usable"][segment_idx, :n].astype(bool)
    idx = np.flatnonzero(mask)
    if idx.size > max_points:
        take = np.linspace(0, idx.size - 1, max_points).round().astype(np.int64)
        idx = idx[take]
    return idx.astype(np.int64)


def _point_ok(point: np.ndarray, width: int, height: int) -> bool:
    return bool(np.isfinite(point).all() and 0 <= point[0] < width and 0 <= point[1] < height)


def _draw_group(
    frame_bgr: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    usable: np.ndarray,
    idx: np.ndarray,
    t: int,
    width: int,
    height: int,
    group: str,
    trail_frames: int,
    draw_non_usable: bool,
) -> None:
    if idx.size == 0:
        return
    start_t = max(0, t - trail_frames + 1)
    color = GROUP_COLORS_BGR[group]
    radius = 2 if group in ("left", "right") else 1
    for j in idx:
        point_color = color if usable[j] else NON_USABLE_COLOR_BGR
        line_color = color if usable[j] else NON_USABLE_COLOR_BGR
        run: list[tuple[int, int]] = []
        for tt in range(start_t, t + 1):
            p = tracks[tt, j]
            if visibility[tt, j] and _point_ok(p, width, height):
                run.append((int(round(float(p[0]))), int(round(float(p[1])))))
            else:
                if len(run) >= 2:
                    cv2.polylines(frame_bgr, [np.asarray(run, dtype=np.int32)], False, line_color, 1, cv2.LINE_AA)
                run = []
        if len(run) >= 2:
            cv2.polylines(frame_bgr, [np.asarray(run, dtype=np.int32)], False, line_color, 1, cv2.LINE_AA)
        p = tracks[t, j]
        if visibility[t, j] and _point_ok(p, width, height):
            cv2.circle(frame_bgr, (int(round(float(p[0]))), int(round(float(p[1])))), radius, point_color, -1, cv2.LINE_AA)


def _ffmpeg_writer(path: Path, width: int, height: int, fps: float, crf: int, preset: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def visualize(args: argparse.Namespace) -> Path:
    npz_path = args.npz_path
    summary_path = args.summary_path or _infer_summary_path(npz_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    with np.load(npz_path, allow_pickle=False) as data:
        source_video = args.source_video or summary.get("source_video_path") or _scalar_str(data["source_video_path"])
        source_fps = float(args.source_fps or summary.get("manifest_fps") or float(data["manifest_fps"]))
        output_fps = float(args.output_fps or float(data["fps"]))
        width = int(data["image_width"])
        height = int(data["image_height"])
        starts = data["source_start_frames"].astype(int)
        lengths = data["segment_lengths"].astype(int)
        start_segment = max(0, int(args.start_segment))
        end_segment = len(starts) if args.end_segment is None else min(len(starts), int(args.end_segment))
        if start_segment >= end_segment:
            raise ValueError(f"empty segment range: {start_segment}:{end_segment}")

        output_path = args.output_path
        if output_path is None:
            stem = npz_path.name[: -len("_scene_tracks.npz")] if npz_path.name.endswith("_scene_tracks.npz") else npz_path.stem
            output_path = npz_path.with_name(f"{stem}_tracks_visualization.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path = args.preview_path or output_path.with_name(output_path.stem + "_preview.jpg")
        tmp_dir = args.tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=output_path.stem + "_", suffix=".mp4", dir=tmp_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        tmp_path.unlink(missing_ok=True)

        proc = _ffmpeg_writer(tmp_path, width, height, output_fps, args.crf, args.preset)
        assert proc.stdin is not None
        frame_count = 0
        preview_saved = False
        try:
            for s in range(start_segment, end_segment):
                seg_len = int(lengths[s])
                if seg_len <= 0:
                    continue
                frames = decode_frames_rgb(str(source_video), int(starts[s]), seg_len, fps=source_fps)
                seg_len = min(seg_len, int(frames.shape[0]))
                if frames.shape[1] != height or frames.shape[2] != width:
                    frames = np.stack(
                        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames[:seg_len]],
                        axis=0,
                    )
                per_group: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                for group in args.groups:
                    key = _track_key(data, group, args.track_source)
                    max_points = args.max_env_points if group == "env" else args.max_robot_points
                    idx = _select_indices(data, group, s, max_points, args.draw_non_usable)
                    if idx.size == 0:
                        continue
                    n = int(data[f"{group}_num_points"][s])
                    tracks = data[key][s, :seg_len, :n]
                    visibility = data[f"{group}_visibility"][s, :seg_len, :n].astype(bool)
                    usable = data[f"{group}_usable"][s, :n].astype(bool)
                    per_group[group] = (tracks, visibility, usable, idx)
                for t in range(seg_len):
                    frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR).copy()
                    for group, (tracks, visibility, usable, idx) in per_group.items():
                        _draw_group(
                            frame,
                            tracks,
                            visibility,
                            usable,
                            idx,
                            t,
                            width,
                            height,
                            group,
                            args.trail_frames,
                            args.draw_non_usable,
                        )
                    if not preview_saved:
                        cv2.imwrite(str(preview_path), frame)
                        preview_saved = True
                    proc.stdin.write(frame.tobytes())
                    frame_count += 1
                if args.progress_interval > 0 and ((s - start_segment + 1) % args.progress_interval == 0 or s + 1 == end_segment):
                    print(f"rendered segment {s + 1}/{end_segment}, frames={frame_count}", flush=True)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")
        shutil.copy2(tmp_path, output_path)
        tmp_path.unlink(missing_ok=True)
        print(f"wrote {output_path}")
        print(f"preview {preview_path}")
        print(f"frames {frame_count}")
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SceneFlowTracker NPZ tracks on the source video.")
    parser.add_argument("npz_path", type=Path, help="Path to *_scene_tracks.npz.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Path to *_summary.json. Defaults to sibling summary.")
    parser.add_argument("--source-video", default=None, help="Override source video path.")
    parser.add_argument("--source-fps", type=float, default=None, help="Override source video fps.")
    parser.add_argument("--output-fps", type=float, default=None, help="Override visualization fps. Defaults to NPZ fps.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output mp4 path.")
    parser.add_argument("--preview-path", type=Path, default=None, help="Output preview jpg path.")
    parser.add_argument("--groups", type=_parse_groups, default=_parse_groups("left,right,env"), help="Comma-separated groups: left,right,env.")
    parser.add_argument("--track-source", choices=("smooth", "raw"), default="smooth", help="Draw smoothed or raw CoTracker tracks.")
    parser.add_argument("--draw-non-usable", action="store_true", help="Also draw non-usable points in gray.")
    parser.add_argument("--trail-frames", type=int, default=30, help="Number of previous frames to draw as a trajectory tail.")
    parser.add_argument("--max-robot-points", type=int, default=1000, help="Max left/right points drawn per segment.")
    parser.add_argument("--max-env-points", type=int, default=120, help="Max environment points drawn per segment.")
    parser.add_argument("--start-segment", type=int, default=0, help="First segment index to visualize.")
    parser.add_argument("--end-segment", type=int, default=None, help="Exclusive end segment index.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp"), help="Local temp dir used for mp4 encoding before copying to output.")
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF. Lower is higher quality.")
    parser.add_argument("--preset", default="veryfast", help="x264 preset.")
    parser.add_argument("--progress-interval", type=int, default=20, help="Print progress every N segments; 0 disables progress.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visualize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
