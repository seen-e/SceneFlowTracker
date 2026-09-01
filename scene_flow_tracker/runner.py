import argparse
import logging

from .config import load_config, validate_config
from .orchestration.pipeline_runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch scene tracking.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--segment-id", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
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
        cfg["models"]["cotracker"]["segment_batch_size"] = 1
    return cfg


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(processName)s %(levelname)s %(message)s")
    cfg = apply_cli_overrides(load_config(args.config), args)
    perf = run(cfg, args.episode_index, args.max_episodes, args.segment_id)
    print(__import__("json").dumps(perf, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
