#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_flow_tracker.config import load_config, validate_config
from scene_flow_tracker.orchestration.pipeline_runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the refactored SceneFlowTracker pipeline.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "config.yaml"))
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--segment-id", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--yolo-batch-size", type=int, default=None)
    parser.add_argument("--cotracker-segment-batch-size", type=int, default=None)
    parser.add_argument("--max-inflight-segments", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_resume:
        cfg["batch"]["resume"] = False
    if args.yolo_batch_size is not None:
        cfg["models"]["yolo"]["batch_size"] = args.yolo_batch_size
    if args.cotracker_segment_batch_size is not None:
        cfg["models"]["cotracker"]["segment_batch_size"] = args.cotracker_segment_batch_size
    if args.max_inflight_segments is not None:
        cfg["pipeline"]["max_inflight_segments"] = args.max_inflight_segments
    validate_config(cfg)
    started = time.perf_counter()
    perf = run(cfg, episode_index=args.episode_index, max_episodes=args.max_episodes, segment_id=args.segment_id)
    perf["benchmark_wall_time_sec"] = time.perf_counter() - started
    perf["config"] = {
        "video": cfg["video"],
        "workers": cfg["workers"],
        "pipeline": cfg["pipeline"],
        "yolo_batch_size": cfg["models"]["yolo"].get("batch_size"),
        "cotracker_segment_batch_size": cfg["models"]["cotracker"].get("segment_batch_size"),
        "total_query_points": cfg["sampling"]["query_allocation"]["total_query_points"],
    }
    text = json.dumps(perf, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
