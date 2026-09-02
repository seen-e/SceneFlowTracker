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
    parser.add_argument("--first-frame-decode-workers", type=int, default=None, help="覆盖首帧解码 worker 数。")
    parser.add_argument("--sampling-workers", type=int, default=None, help="覆盖采样 worker 数。")
    parser.add_argument("--segment-decode-workers", type=int, default=None, help="覆盖完整 segment 解码 worker 数。")
    parser.add_argument("--filter-workers", type=int, default=None, help="覆盖轨迹过滤 worker 数。")
    parser.add_argument("--yolo-batch-size", type=int, default=None, help="覆盖 YOLO 首帧 batch size。")
    parser.add_argument("--yolo-worker-count", type=int, default=None, help="覆盖 YOLO 模型 worker 数量。")
    parser.add_argument("--yolo-devices", default=None, help="覆盖 YOLO GPU 列表，逗号分隔，例如 cuda:0,cuda:1。")
    parser.add_argument("--cotracker-segment-batch-size", type=int, default=None, help="覆盖 CoTracker segment batch size。")
    parser.add_argument("--cotracker-worker-count", type=int, default=None, help="覆盖 CoTracker 模型 worker 数量。")
    parser.add_argument("--cotracker-devices", default=None, help="覆盖 CoTracker GPU 列表，逗号分隔，例如 cuda:0,cuda:1。")
    parser.add_argument("--total-query-points", type=int, default=None, help="覆盖每个 segment 固定 query 点总数。")
    parser.add_argument("--decode-workers", type=int, default=None, help="兼容旧参数：同时覆盖首帧解码和完整 segment 解码 worker 数。")
    parser.add_argument("--model-workers", type=int, default=None, help="兼容旧参数：映射为 CoTracker 模型 worker 数量。")
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
    if args.first_frame_decode_workers is not None:
        cfg["workers"]["first_frame_decode_workers"] = args.first_frame_decode_workers
    if args.sampling_workers is not None:
        cfg["workers"]["sampling_workers"] = args.sampling_workers
    if args.segment_decode_workers is not None:
        cfg["workers"]["segment_decode_workers"] = args.segment_decode_workers
    if args.filter_workers is not None:
        cfg["workers"]["filter_workers"] = args.filter_workers
    if args.yolo_batch_size is not None:
        cfg["models"]["yolo"]["batch_size"] = args.yolo_batch_size
    if args.yolo_worker_count is not None:
        cfg["models"]["yolo"]["worker_count"] = args.yolo_worker_count
    if args.yolo_devices:
        cfg["models"]["yolo"]["devices"] = [item.strip() for item in args.yolo_devices.split(",") if item.strip()]
    if args.cotracker_segment_batch_size is not None:
        cfg["models"]["cotracker"]["segment_batch_size"] = args.cotracker_segment_batch_size
    if args.cotracker_worker_count is not None:
        cfg["models"]["cotracker"]["worker_count"] = args.cotracker_worker_count
    if args.cotracker_devices:
        cfg["models"]["cotracker"]["devices"] = [item.strip() for item in args.cotracker_devices.split(",") if item.strip()]
    if args.total_query_points is not None:
        cfg["sampling"]["query_allocation"]["total_query_points"] = args.total_query_points
    if args.decode_workers is not None:
        cfg["workers"]["first_frame_decode_workers"] = args.decode_workers
        cfg["workers"]["segment_decode_workers"] = args.decode_workers
    if args.model_workers is not None:
        cfg["models"]["cotracker"]["worker_count"] = args.model_workers
    if args.resume:
        cfg["batch"]["resume"] = True
    if args.no_resume:
        cfg["batch"]["resume"] = False
    if args.debug:
        cfg["output"]["debug_visualization"] = True
        cfg["workers"]["first_frame_decode_workers"] = 1
        cfg["workers"]["sampling_workers"] = 1
        cfg["workers"]["segment_decode_workers"] = 1
        cfg["workers"]["filter_workers"] = 1
        cfg["models"]["yolo"]["batch_size"] = 1
        cfg["models"]["yolo"]["worker_count"] = 1
        cfg["models"]["cotracker"]["segment_batch_size"] = 1
        cfg["models"]["cotracker"]["worker_count"] = 1
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
