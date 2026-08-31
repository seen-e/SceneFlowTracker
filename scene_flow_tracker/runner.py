from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path

from .config import load_config, validate_config
from .manifest import load_episode_jobs
from .pipeline.episode_aggregator import EpisodeAggregator
from .queues import make_context, make_queues
from .segment_planner import plan_segments
from .storage.resume import scan_completed_episode_views
from .storage.writers import append_processing_manifest, append_processing_manifest_many
from .workers.decode_worker import decode_worker
from .workers.model_worker import model_worker


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
        cfg["workers"]["decode_workers"] = 1
        cfg["workers"]["model_workers"] = 1
        cfg["queues"]["decoded_segment_queue_size"] = 2
    return cfg


def select_episodes(episodes, args):
    if args.episode_index is not None:
        episodes = [ep for ep in episodes if ep.episode_index == args.episode_index]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    return episodes


def run(cfg: dict, episode_index: int | None = None, max_episodes: int | None = None, segment_id: int | None = None) -> dict:
    validate_config(cfg)
    output_root = Path(cfg["output"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    episodes, invalid = load_episode_jobs(cfg)
    skipped = []
    if episode_index is not None:
        episodes = [ep for ep in episodes if ep.episode_index == episode_index]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    manifest_path = output_root / "processing_manifest.jsonl"
    manifest_cfg = cfg.get("processing_manifest", {})
    write_skip_records = bool(manifest_cfg.get("write_resume_skip_records", False))
    write_running_records = bool(manifest_cfg.get("write_running_records", False))
    if cfg["batch"].get("resume", True):
        scan = scan_completed_episode_views(output_root, episodes)
        skipped = [ep for ep in episodes if scan.is_completed(ep)]
        if skipped and write_skip_records:
            append_processing_manifest_many(
                manifest_path,
                [
                    {
                        "dataset": ep.dataset,
                        "episode_id": ep.episode_id,
                        "episode_index": ep.episode_index,
                        "view": ep.view_key,
                        "status": "SKIPPED_RESUME",
                    }
                    for ep in skipped
                ],
            )
        episodes = [ep for ep in episodes if ep not in skipped]
        logging.info(
            "Resume scan: completed=%d selected=%d existing_episode_dirs=%d selected_existing_dirs=%d npz=%d missing_summary=%d elapsed=%.2fs",
            len(skipped),
            len(skipped) + len(episodes),
            scan.existing_episode_dirs,
            scan.selected_existing_dirs,
            scan.existing_npz_files,
            scan.missing_summary_files,
            scan.elapsed_sec,
        )
    planning_started = time.perf_counter()
    expected_counts = {}
    total_segments = 0
    total_video_seconds = 0.0
    running_manifest_items = []
    for ep in episodes:
        segs = plan_segments(ep, int(cfg["video"]["segment_frames"]), cfg["video"]["tail_policy"])
        if segment_id is not None:
            segs = [s for s in segs if s.segment_id == segment_id]
        count = len(segs)
        expected_counts[ep.episode_id] = count
        total_segments += count
        total_video_seconds += sum(s.frame_count / s.effective_fps for s in segs)
        running_manifest_items.append({"dataset": ep.dataset, "episode_id": ep.episode_id, "episode_index": ep.episode_index, "view": ep.view_key, "video_path": ep.physical_video_path, "source_start_frame": ep.source_start_frame, "source_end_frame": ep.source_end_frame, "status": "RUNNING", "num_segments": count})
    if running_manifest_items and write_running_records:
        append_processing_manifest_many(manifest_path, running_manifest_items)
    elif skipped or running_manifest_items:
        append_processing_manifest(
            manifest_path,
            {
                "status": "BATCH_PLANNED",
                "resume_skipped_episodes": len(skipped) if cfg["batch"].get("resume", True) else 0,
                "running_episodes": len(running_manifest_items),
                "running_segments": total_segments,
            },
        )
    logging.info("Planned batch: episodes=%d segments=%d manifest_items=%d elapsed=%.2fs", len(episodes), total_segments, len(running_manifest_items), time.perf_counter() - planning_started)
    if total_segments == 0:
        return {"episodes_processed": 0, "segments_processed": 0, "invalid_manifest_items": len(invalid)}

    started = time.perf_counter()
    logging.info("Starting batch: episodes=%d segments=%d output_root=%s", len(episodes), total_segments, output_root)
    ctx = make_context("spawn")
    job_queue, decoded_queue, result_queue = make_queues(ctx, cfg)
    log_file = cfg.get("_log_file")
    log_level = cfg.get("_log_level", "INFO")
    decoders = [
        ctx.Process(target=decode_worker, args=(i, job_queue, decoded_queue, result_queue, bool(cfg["batch"]["continue_on_segment_error"]), log_file, log_level))
        for i in range(int(cfg["workers"]["decode_workers"]))
    ]
    devices = list(cfg["workers"].get("model_devices") or ["cpu"])
    models = [
        ctx.Process(target=model_worker, args=(i, cfg, devices[i % len(devices)], decoded_queue, result_queue))
        for i in range(int(cfg["workers"]["model_workers"]))
    ]

    def feed_segment_jobs() -> None:
        try:
            for ep in episodes:
                segs = plan_segments(ep, int(cfg["video"]["segment_frames"]), cfg["video"]["tail_policy"])
                if segment_id is not None:
                    segs = [s for s in segs if s.segment_id == segment_id]
                for seg in segs:
                    job_queue.put(seg)
        finally:
            for _ in decoders:
                job_queue.put(None)

    for proc in decoders + models:
        proc.start()
    feeder = threading.Thread(target=feed_segment_jobs, name="segment-feeder", daemon=True)
    feeder.start()

    aggregator = EpisodeAggregator(episodes, expected_counts, output_root, bool(cfg["batch"]["atomic_write"]), cfg)
    completed = failed = model_sentinels = 0
    timing_acc: dict[str, list[float]] = {}
    model_sentinels_sent = False
    while completed + failed < total_segments:
        if not model_sentinels_sent and all(not p.is_alive() for p in decoders):
            for proc in decoders:
                if proc.exitcode not in (0, None):
                    raise RuntimeError(f"decode worker exited unexpectedly: {proc.exitcode}")
            for _ in models:
                decoded_queue.put(None)
            model_sentinels_sent = True
        try:
            item = result_queue.get(timeout=30)
        except queue.Empty:
            dead = [p.exitcode for p in decoders + models if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"worker exited unexpectedly: {dead}")
            continue
        if item is None:
            model_sentinels += 1
            continue
        if item.status == "DONE":
            completed += 1
        else:
            failed += 1
        for key, value in item.timings.items():
            timing_acc.setdefault(key, []).append(float(value))
        aggregator.add(item)

    for proc in decoders:
        proc.join()
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"decode worker exited unexpectedly: {proc.exitcode}")
    if not model_sentinels_sent:
        for _ in models:
            decoded_queue.put(None)
        model_sentinels_sent = True
    for proc in models:
        proc.join()
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"model worker exited unexpectedly: {proc.exitcode}")
    elapsed = time.perf_counter() - started
    def avg(name: str) -> float:
        vals = timing_acc.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0
    perf = {
        "episodes_processed": len(episodes),
        "episodes_failed": sum(1 for ep in episodes if any(r.status != "DONE" for r in aggregator.results.get(ep.episode_id, []))),
        "segments_processed": completed,
        "segments_failed": failed,
        "average_decode_time": avg("decode_time_sec"),
        "average_yolo_time": avg("yolo_time_sec"),
        "average_cotracker_time": avg("cotracker_time_sec"),
        "average_segment_processing_time": avg("segment_total_time_sec"),
        "average_episode_time_sec": elapsed / max(1, len(episodes)),
        "episodes_per_hour": len(episodes) / max(1e-6, elapsed) * 3600.0,
        "total_wall_time": elapsed,
        "segments_per_second": total_segments / max(1e-6, elapsed),
        "video_seconds_per_wall_second": total_video_seconds / max(1e-6, elapsed),
    }
    logging.info(
        "Batch complete: episodes=%d failed=%d segments=%d failed_segments=%d avg_episode_sec=%.2f episodes_per_hour=%.2f total_wall_time=%.2f",
        perf["episodes_processed"],
        perf["episodes_failed"],
        perf["segments_processed"],
        perf["segments_failed"],
        perf["average_episode_time_sec"],
        perf["episodes_per_hour"],
        perf["total_wall_time"],
    )
    (output_root / "performance_summary.json").write_text(__import__("json").dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    return perf


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(processName)s %(levelname)s %(message)s")
    cfg = apply_cli_overrides(load_config(args.config), args)
    perf = run(cfg, args.episode_index, args.max_episodes, args.segment_id)
    print(__import__("json").dumps(perf, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
