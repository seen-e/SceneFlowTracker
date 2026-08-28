#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_flow_tracker.config import load_config, validate_config
from scene_flow_tracker.runner import run


DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_MANIFEST = "/mnt/data/chachaxu/save/abc_130k_v3/abc_130k_v3_train_all_views.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SceneFlowTracker example entrypoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="基础配置文件路径。")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST, help="ABC-130K all views manifest 路径。")
    parser.add_argument("--view-key", default=None, help="指定处理视角，例如 observation.images.top。")
    parser.add_argument("--output-root", default=None, help="覆盖输出根目录。")
    parser.add_argument("--segment-frames", type=int, default=None, help="覆盖每个 segment 的帧数。")
    parser.add_argument("--decode-workers", type=int, default=None, help="覆盖视频解码 worker 数。")
    parser.add_argument("--model-workers", type=int, default=None, help="覆盖模型 worker 数。")
    parser.add_argument("--episode-index", type=int, default=None, help="只处理指定 episode_index。")
    parser.add_argument("--max-episodes", type=int, default=None, help="最多处理多少个 episode。")
    parser.add_argument("--segment-id", type=int, default=None, help="只处理指定 segment_id，通常用于 smoke test。")
    parser.add_argument("--resume", action="store_true", default=None, help="跳过已完成 episode。")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已完成 episode。")
    parser.add_argument("--debug", action="store_true", help="调试模式：降低 worker 并打开 debug 日志。")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg["input"]["manifest_path"] = args.manifest_path
    if args.view_key:
        cfg["input"]["view_key"] = args.view_key
    if args.output_root:
        cfg["output"]["output_root"] = args.output_root
    if args.segment_frames is not None:
        cfg["video"]["segment_frames"] = args.segment_frames
    if args.decode_workers is not None:
        cfg["workers"]["decode_workers"] = args.decode_workers
    if args.model_workers is not None:
        cfg["workers"]["model_workers"] = args.model_workers
    if args.resume:
        cfg["batch"]["resume"] = True
    if args.no_resume:
        cfg["batch"]["resume"] = False
    if args.debug:
        cfg["output"]["debug_visualization"] = True
        cfg["workers"]["decode_workers"] = 1
        cfg["workers"]["model_workers"] = 1
        cfg["queues"]["decoded_segment_queue_size"] = 2
    validate_config(cfg)
    return cfg


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    cfg = build_config(args)
    perf = run(
        cfg,
        episode_index=args.episode_index,
        max_episodes=args.max_episodes,
        segment_id=args.segment_id,
    )
    print(json.dumps(perf, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
