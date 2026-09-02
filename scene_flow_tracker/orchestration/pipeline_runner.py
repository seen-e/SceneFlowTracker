from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from ..algorithms.edges import detect_content_bbox
from ..config import validate_config
from ..data.types import CoTrackerBatch, DecodedTrackItem, FilteredSegmentResult, FirstFrameItem, SamplingResult, YoloDetectionResult
from ..inference.cotracker_model import CoTrackerModel
from ..inference.yolo_model import YoloModel
from ..jobs import EpisodeJob, SegmentJob, SegmentResult
from ..manifest import load_episode_jobs
from ..model_parallel import expanded_worker_devices
from ..pipeline.result_builder import filtered_failure, filtered_to_segment_result
from ..segment_planner import plan_segments
from ..storage.resume import scan_completed_episode_views
from ..storage.segment_cache import (
    config_fingerprint,
    delete_segment_cache_dir,
    read_segment_cache,
    segment_cache_path,
    validate_segment_cache,
    write_segment_cache,
)
from ..storage.writers import append_processing_manifest, append_processing_manifest_many, write_episode_outputs
from ..workers.first_frame_decode_worker import decode_first_frame
from ..workers.filter_worker import filter_track_result
from ..workers.sampling_worker import sample_queries
from ..workers.segment_decode_worker import decode_segment_rgb
from ..utils.shared_arrays import create_shared_array
from ..utils.shared_arrays import release_shared_array
from ..video_decode import decode_first_frame_rgb


def _pool_map(
    name: str,
    workers: int,
    fn: Callable[[Any], Any],
    items: list[Any],
    on_failure: Callable[[Any, Exception], Any] | None = None,
) -> list[Any]:
    if not items:
        return []
    workers = max(1, int(workers))
    out: list[Any] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                out.append(fut.result())
            except Exception as exc:
                if on_failure is None:
                    raise
                out.append(on_failure(item, exc))
    return out


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _filter_episodes(episodes: list[EpisodeJob], episode_index: int | None, max_episodes: int | None) -> list[EpisodeJob]:
    if episode_index is not None:
        episodes = [ep for ep in episodes if ep.episode_index == episode_index]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def _planned_segments(episode: EpisodeJob, cfg: dict[str, Any], segment_id: int | None) -> list[SegmentJob]:
    segments = plan_segments(episode, int(cfg["video"]["segment_frames"]), str(cfg["video"]["tail_policy"]))
    if segment_id is not None:
        segments = [seg for seg in segments if seg.segment_id == segment_id]
    return segments


class PipelineRunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_root = Path(cfg["output"]["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = config_fingerprint(cfg)
        self.timing_acc: dict[str, list[float]] = {}
        self.yolo_models: list[YoloModel] | None = None
        self.cotracker_models: list[CoTrackerModel] | None = None

    def _record_timings(self, result: SegmentResult) -> None:
        for key, value in result.timings.items():
            try:
                self.timing_acc.setdefault(key, []).append(float(value))
            except Exception:
                continue

    def _get_yolo_workers(self) -> list[YoloModel]:
        if self.yolo_models is None:
            devices = expanded_worker_devices(self.cfg["models"]["yolo"], "cuda:0")
            logging.info("loading %d YOLO worker(s) on devices=%s", len(devices), devices)
            self.yolo_models = [YoloModel(self.cfg, device=device, worker_id=idx) for idx, device in enumerate(devices)]
        return self.yolo_models

    def _get_cotracker_workers(self) -> list[CoTrackerModel]:
        if self.cotracker_models is None:
            devices = expanded_worker_devices(self.cfg["models"]["cotracker"], "cuda:0")
            logging.info("loading %d CoTracker worker(s) on devices=%s", len(devices), devices)
            self.cotracker_models = [CoTrackerModel(self.cfg, device=device, worker_id=idx) for idx, device in enumerate(devices)]
        return self.cotracker_models

    def _process_first_frames(self, segments: list[SegmentJob]) -> tuple[list[FirstFrameItem], list[SegmentResult]]:
        def work(job: SegmentJob) -> FirstFrameItem:
            started = time.perf_counter()
            return FirstFrameItem(job=job, frame_rgb=decode_first_frame(job), timings={"first_frame_decode_time_sec": time.perf_counter() - started})

        def fail(job: SegmentJob, exc: Exception) -> SegmentResult:
            logging.exception("first-frame decode failed episode=%s segment=%s", job.episode_id, job.segment_id)
            return filtered_to_segment_result(filtered_failure(job, type(exc).__name__, str(exc)))

        items = _pool_map("first-frame-decode", int(self.cfg["workers"]["first_frame_decode_workers"]), work, segments, fail)
        ok = [item for item in items if isinstance(item, FirstFrameItem)]
        failed = [item for item in items if isinstance(item, SegmentResult)]
        return ok, failed

    def _process_yolo(self, items: list[FirstFrameItem]) -> list[YoloDetectionResult]:
        models = self._get_yolo_workers()
        ycfg = self.cfg["models"]["yolo"]
        batch_size = int(ycfg.get("batch_size", 32))
        batches = list(_chunks(items, batch_size))
        partitions: list[list[list[FirstFrameItem]]] = [[] for _ in models]
        for idx, batch in enumerate(batches):
            partitions[idx % len(models)].append(batch)

        def work(worker_idx: int) -> list[YoloDetectionResult]:
            model = models[worker_idx]
            worker_results: list[YoloDetectionResult] = []
            for batch in partitions[worker_idx]:
                batch_results, elapsed = model.predict_batch(batch)
                for result in batch_results:
                    result.timings["yolo_batch_size"] = float(len(batch))
                    result.timings["yolo_batch_fill_ratio"] = float(len(batch) / max(1, batch_size))
                    result.timings["yolo_forward_time_sec"] = elapsed
                    result.timings["yolo_model_worker_count"] = float(len(models))
                worker_results.extend(batch_results)
            return worker_results

        if len(models) == 1 or len(batches) <= 1:
            results = work(0)
        else:
            results = []
            with ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="yolo-model") as pool:
                futures = [pool.submit(work, idx) for idx in range(len(models)) if partitions[idx]]
                for fut in as_completed(futures):
                    results.extend(fut.result())
        return sorted(results, key=lambda x: (x.job.episode_index, x.job.segment_id))

    def _process_sampling(self, items: list[YoloDetectionResult]) -> tuple[list[SamplingResult], list[SegmentResult]]:
        def fail(item: YoloDetectionResult, exc: Exception) -> SegmentResult:
            logging.exception("sampling failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            return filtered_to_segment_result(filtered_failure(item.job, type(exc).__name__, str(exc), dict(item.timings)))

        out = _pool_map("sampling", int(self.cfg["workers"]["sampling_workers"]), lambda item: sample_queries(item, self.cfg), items, fail)
        return [x for x in out if isinstance(x, SamplingResult)], [x for x in out if isinstance(x, SegmentResult)]

    def _process_segment_decode(self, items: list[SamplingResult]) -> tuple[list[DecodedTrackItem], list[SegmentResult]]:
        def work(item: SamplingResult) -> DecodedTrackItem:
            started = time.perf_counter()
            frames = decode_segment_rgb(item.job)
            ref = create_shared_array(frames, owner="pipeline_runner", debug_id=f"{item.job.episode_id}:{item.job.segment_id}")
            timings = dict(item.timings)
            timings["segment_decode_time_sec"] = time.perf_counter() - started
            return DecodedTrackItem(
                job=item.job,
                frame_ref=ref,
                query_xy=item.query_xy,
                query_group=item.query_group,
                detections=item.detections,
                sampling_features=item.sampling_features,
                sampling_stats=item.sampling_stats,
                image_height=item.detections.image_height,
                image_width=item.detections.image_width,
                timings=timings,
            )

        def fail(item: SamplingResult, exc: Exception) -> SegmentResult:
            logging.exception("segment decode failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            return filtered_to_segment_result(filtered_failure(item.job, type(exc).__name__, str(exc), dict(item.timings)))

        out = _pool_map("segment-decode", int(self.cfg["workers"]["segment_decode_workers"]), work, items, fail)
        return [x for x in out if isinstance(x, DecodedTrackItem)], [x for x in out if isinstance(x, SegmentResult)]

    def _process_cotracker(self, items: list[DecodedTrackItem]) -> list:
        models = self._get_cotracker_workers()
        ccfg = self.cfg["models"]["cotracker"]
        batch_size = int(ccfg.get("segment_batch_size", 4))
        normal_t = int(self.cfg["video"]["segment_frames"])
        groups: dict[tuple[int, int, int, int], list[DecodedTrackItem]] = {}
        batches: list[CoTrackerBatch] = []
        for item in sorted(items, key=lambda x: (x.job.frame_count != normal_t, x.job.episode_index, x.job.segment_id)):
            key = (item.job.frame_count, item.image_height, item.image_width, int(item.query_xy.shape[0]))
            if item.job.frame_count != normal_t:
                batch = CoTrackerBatch(items=[item], batch_key=key, batch_size=1, fill_ratio=1.0 / max(1, batch_size), is_tail=True)
                batches.append(batch)
            else:
                bucket = groups.setdefault(key, [])
                bucket.append(item)
                if len(bucket) >= batch_size:
                    take = bucket[:batch_size]
                    del bucket[:batch_size]
                    batch = CoTrackerBatch(items=take, batch_key=key, batch_size=len(take), fill_ratio=1.0, is_tail=False)
                    batches.append(batch)
        for key, bucket in groups.items():
            if not bucket:
                continue
            batch = CoTrackerBatch(
                items=list(bucket),
                batch_key=key,
                batch_size=len(bucket),
                fill_ratio=len(bucket) / max(1, batch_size),
                is_tail=False,
            )
            batches.append(batch)

        partitions: list[list[CoTrackerBatch]] = [[] for _ in models]
        for idx, batch in enumerate(batches):
            partitions[idx % len(models)].append(batch)

        def work(worker_idx: int) -> list:
            model = models[worker_idx]
            worker_results = []
            for batch in partitions[worker_idx]:
                results, _elapsed, peak = model.track_batch(batch)
                for result in results:
                    result.timings["cotracker_model_worker_count"] = float(len(models))
                    if peak is not None:
                        result.timings["cotracker_gpu_peak_memory_mb"] = float(peak)
                worker_results.extend(results)
            return worker_results

        if len(models) == 1 or len(batches) <= 1:
            track_results = work(0) if batches else []
        else:
            track_results = []
            with ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="cotracker-model") as pool:
                futures = [pool.submit(work, idx) for idx in range(len(models)) if partitions[idx]]
                for fut in as_completed(futures):
                    track_results.extend(fut.result())
        return sorted(track_results, key=lambda x: (x.job.episode_index, x.job.segment_id))

    def _process_filter(self, items: list[Any]) -> list[SegmentResult]:
        def fail(item: Any, exc: Exception) -> SegmentResult:
            logging.exception("filter failed episode=%s segment=%s", item.job.episode_id, item.job.segment_id)
            return filtered_to_segment_result(filtered_failure(item.job, type(exc).__name__, str(exc), dict(getattr(item, "timings", {}))))

        filtered = _pool_map("filter", int(self.cfg["workers"]["filter_workers"]), lambda item: filter_track_result(item, self.cfg), items, fail)
        return [filtered_to_segment_result(item) if isinstance(item, FilteredSegmentResult) else item for item in filtered]

    def _estimate_episode_content_bbox(self, episode: EpisodeJob) -> tuple[int, int, int, int] | None:
        valid_region_cfg = (self.cfg.get("sampling", {}) or {}).get("valid_region", {}) or {}
        if not bool(valid_region_cfg.get("enabled", True)):
            return None
        try:
            fps = episode.manifest_fps or episode.effective_fps
            frame = decode_first_frame_rgb(episode.physical_video_path, episode.source_start_frame, fps=fps)
            bbox = detect_content_bbox(frame, valid_region_cfg)
            logging.info("episode=%s content_bbox_xyxy=%s", episode.episode_id, bbox)
            return bbox
        except Exception as exc:
            logging.warning("content bbox estimation failed episode=%s error=%s", episode.episode_id, exc)
            return None

    def _attach_episode_content_bbox(self, episode: EpisodeJob, segments: list[SegmentJob]) -> list[SegmentJob]:
        bbox = self._estimate_episode_content_bbox(episode)
        if bbox is None:
            return segments
        return [replace(seg, content_bbox_xyxy=bbox) for seg in segments]

    def _process_uncached_segments(self, episode: EpisodeJob, segments: list[SegmentJob]) -> list[SegmentResult]:
        max_inflight = int((self.cfg.get("pipeline", {}) or {}).get("max_inflight_segments", 64))
        all_results: list[SegmentResult] = []
        for chunk in _chunks(segments, max_inflight):
            logging.info("processing chunk episode=%s segments=%s..%s count=%d", episode.episode_id, chunk[0].segment_id, chunk[-1].segment_id, len(chunk))
            first_frames, failed = self._process_first_frames(chunk)
            results = list(failed)
            yolo = self._process_yolo(first_frames) if first_frames else []
            sampled, failed = self._process_sampling(yolo)
            results.extend(failed)
            decoded, failed = self._process_segment_decode(sampled)
            results.extend(failed)
            try:
                tracks = self._process_cotracker(decoded) if decoded else []
                results.extend(self._process_filter(tracks))
            except Exception as exc:
                logging.exception("cotracker stage failed for chunk episode=%s", episode.episode_id)
                for item in decoded:
                    release_shared_array(item.frame_ref, unlink=True)
                    results.append(filtered_to_segment_result(filtered_failure(item.job, type(exc).__name__, str(exc), dict(item.timings))))
            for result in results:
                cache_path = segment_cache_path(self.output_root, episode, result.job, self.cfg)
                write_segment_cache(cache_path, result, self.fingerprint, atomic=True)
                self._record_timings(result)
            all_results.extend(results)
        return all_results

    def _load_valid_caches(self, episode: EpisodeJob, segments: list[SegmentJob]) -> tuple[list[SegmentResult], list[SegmentJob]]:
        if not bool((self.cfg.get("cache", {}) or {}).get("enabled", True)):
            return [], segments
        cached: list[SegmentResult] = []
        missing: list[SegmentJob] = []
        retry_failed = bool((self.cfg.get("cache", {}) or {}).get("retry_cached_failed_segments", False))
        for seg in segments:
            path = segment_cache_path(self.output_root, episode, seg, self.cfg)
            if validate_segment_cache(path, seg, self.fingerprint):
                result, _fp = read_segment_cache(path)
                if retry_failed and result.status != "DONE":
                    missing.append(seg)
                else:
                    cached.append(result)
            else:
                missing.append(seg)
        return cached, missing

    def run_episode(self, episode: EpisodeJob, segment_id: int | None = None) -> dict[str, Any]:
        segments = _planned_segments(episode, self.cfg, segment_id)
        if not segments:
            return {"episode_id": episode.episode_id, "status": "NO_SEGMENTS", "segments": 0}
        segments = self._attach_episode_content_bbox(episode, segments)
        cached, todo = self._load_valid_caches(episode, segments)
        logging.info("episode=%s planned=%d cached=%d todo=%d", episode.episode_id, len(segments), len(cached), len(todo))
        self._process_uncached_segments(episode, todo)
        merged: list[SegmentResult] = []
        for seg in segments:
            path = segment_cache_path(self.output_root, episode, seg, self.cfg)
            if validate_segment_cache(path, seg, self.fingerprint):
                result, _fp = read_segment_cache(path)
                merged.append(result)
            else:
                merged.append(SegmentResult(job=seg, status="FAILED", error_code="MISSING_SEGMENT_CACHE", error_message=str(path)))
        summary = write_episode_outputs(self.output_root, episode, merged, atomic=bool(self.cfg["batch"].get("atomic_write", True)), cfg=self.cfg)
        if bool((self.cfg.get("cache", {}) or {}).get("delete_after_successful_merge", True)):
            delete_segment_cache_dir(self.output_root, episode, self.cfg)
        return summary

    def avg(self, name: str) -> float:
        vals = self.timing_acc.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0


def run(cfg: dict[str, Any], episode_index: int | None = None, max_episodes: int | None = None, segment_id: int | None = None) -> dict[str, Any]:
    validate_config(cfg)
    started = time.perf_counter()
    output_root = Path(cfg["output"]["output_root"])
    episodes, invalid = load_episode_jobs(cfg)
    episodes = _filter_episodes(episodes, episode_index, max_episodes)
    skipped: list[EpisodeJob] = []
    manifest_path = output_root / "processing_manifest.jsonl"
    if cfg["batch"].get("resume", True) and segment_id is None:
        scan = scan_completed_episode_views(output_root, episodes)
        skipped = [ep for ep in episodes if scan.is_completed(ep)]
        episodes = [ep for ep in episodes if ep not in skipped]
        if skipped and bool((cfg.get("processing_manifest", {}) or {}).get("write_resume_skip_records", False)):
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
        logging.info("resume scan skipped=%d remaining=%d elapsed=%.2fs", len(skipped), len(episodes), scan.elapsed_sec)
    total_segments = sum(len(_planned_segments(ep, cfg, segment_id)) for ep in episodes)
    append_processing_manifest(
        manifest_path,
        {
            "status": "BATCH_PLANNED",
            "resume_skipped_episodes": len(skipped),
            "running_episodes": len(episodes),
            "running_segments": total_segments,
            "pipeline": "refactored_batch_stages",
        },
    )
    if not episodes or total_segments == 0:
        return {"episodes_processed": 0, "segments_processed": 0, "invalid_manifest_items": len(invalid)}
    runner = PipelineRunner(cfg)
    summaries: list[dict[str, Any]] = []
    for ep in episodes:
        summaries.append(runner.run_episode(ep, segment_id=segment_id))
    elapsed = time.perf_counter() - started
    segments_failed = sum(int(s.get("segments_failed", 0)) for s in summaries)
    perf = {
        "pipeline": "refactored_batch_stages",
        "episodes_processed": len(episodes),
        "episodes_failed": sum(1 for s in summaries if int(s.get("segments_failed", 0)) > 0),
        "segments_processed": total_segments - segments_failed,
        "segments_failed": segments_failed,
        "invalid_manifest_items": len(invalid),
        "resume_skipped_episodes": len(skipped),
        "average_first_frame_decode_time": runner.avg("first_frame_decode_time_sec"),
        "average_yolo_time": runner.avg("yolo_time_sec"),
        "average_sampling_time": runner.avg("sampling_time_sec"),
        "average_segment_decode_time": runner.avg("segment_decode_time_sec"),
        "average_cotracker_time": runner.avg("cotracker_time_sec"),
        "average_filter_time": runner.avg("filter_time_sec"),
        "total_wall_time": elapsed,
        "segments_per_second": total_segments / max(1e-6, elapsed),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "performance_summary.json").write_text(json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("batch complete episodes=%d segments=%d failed=%d elapsed=%.2fs", len(episodes), total_segments, segments_failed, elapsed)
    return perf
