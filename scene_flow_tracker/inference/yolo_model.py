from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..data.types import FirstFrameItem, YoloDetectionResult


def _as_xyxy(value: Any, width: int, height: int) -> np.ndarray:
    box = np.asarray(value, dtype=np.float32).reshape(4)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    if box[2] < box[0]:
        box[0], box[2] = box[2], box[0]
    if box[3] < box[1]:
        box[1], box[3] = box[3], box[1]
    return box


def assign_arm_slots(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    names: dict[int, str] | None = None,
) -> tuple[np.ndarray | None, float, np.ndarray | None, float, str]:
    candidates: list[dict[str, Any]] = []
    for det in detections:
        box = _as_xyxy(det["bbox_xyxy"], image_width, image_height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        item = dict(det)
        item["bbox_xyxy"] = box
        item["confidence"] = float(det.get("confidence", np.nan))
        candidates.append(item)
    if not candidates:
        return None, float("nan"), None, float("nan"), "none"

    left = None
    right = None
    method = "x_center"
    if names:
        for item in sorted(candidates, key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
            class_name = str(names.get(int(item.get("class_id", -1)), "")).lower()
            if "left" in class_name and left is None:
                left = item
                method = "class_name"
            elif "right" in class_name and right is None:
                right = item
                method = "class_name"

    remaining = [item for item in candidates if item is not left and item is not right]
    if left is None or right is None:
        if len(remaining) == 1 and left is None and right is None:
            cx = float((remaining[0]["bbox_xyxy"][0] + remaining[0]["bbox_xyxy"][2]) * 0.5)
            if cx < image_width * 0.5:
                left = remaining[0]
            else:
                right = remaining[0]
        else:
            ordered = sorted(remaining, key=lambda d: float((d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) * 0.5))
            if left is None and ordered:
                left = ordered[0]
            if right is None and len(ordered) >= 2:
                right = ordered[-1]
    left_box = None if left is None else np.asarray(left["bbox_xyxy"], dtype=np.float32)
    right_box = None if right is None else np.asarray(right["bbox_xyxy"], dtype=np.float32)
    left_conf = float("nan") if left is None else float(left.get("confidence", np.nan))
    right_conf = float("nan") if right is None else float(right.get("confidence", np.nan))
    return left_box, left_conf, right_box, right_conf, method


class YoloModel:
    def __init__(self, cfg: dict[str, Any], device: str | None = None, worker_id: int = 0) -> None:
        from ultralytics import YOLO

        ycfg = cfg["models"]["yolo"]
        self.model = YOLO(str(ycfg["model_path"]))
        self.device = str(device or ycfg.get("device", "cuda:0"))
        self.worker_id = int(worker_id)
        self.imgsz = int(ycfg.get("imgsz", 640))
        self.conf = float(ycfg.get("conf", ycfg.get("confidence_threshold", 0.25)))
        self.iou = float(ycfg.get("iou", 0.7))
        self.names = getattr(self.model, "names", None)

    def predict_batch(self, items: list[FirstFrameItem]) -> tuple[list[YoloDetectionResult], float]:
        started = time.perf_counter()
        frames = [item.frame_rgb for item in items]
        outputs = self.model.predict(
            source=frames,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        results: list[YoloDetectionResult] = []
        for item, pred in zip(items, outputs):
            h, w = item.frame_rgb.shape[:2]
            raw: list[dict[str, Any]] = []
            boxes = getattr(pred, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy()
                for i in range(len(xyxy)):
                    raw.append(
                        {
                            "bbox_xyxy": np.asarray(xyxy[i], dtype=np.float32).tolist(),
                            "confidence": float(conf[i]),
                            "class_id": int(cls[i]),
                        }
                    )
            left_box, left_conf, right_box, right_conf, method = assign_arm_slots(raw, w, h, self.names)
            timings = dict(item.timings)
            timings["yolo_time_sec"] = elapsed / max(1, len(items))
            timings["yolo_worker_id"] = float(self.worker_id)
            results.append(
                YoloDetectionResult(
                    job=item.job,
                    first_frame_rgb=item.frame_rgb,
                    left_bbox_xyxy=left_box,
                    left_bbox_valid=left_box is not None,
                    left_confidence=left_conf,
                    right_bbox_xyxy=right_box,
                    right_bbox_valid=right_box is not None,
                    right_confidence=right_conf,
                    raw_detections=raw,
                    assignment_method=method,
                    image_height=h,
                    image_width=w,
                    timings=timings,
                )
            )
        return results, elapsed
