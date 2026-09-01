from __future__ import annotations

import cv2
import numpy as np

from .edges import grayscale


def trackability_map(image_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    gray = grayscale(image_rgb).astype(np.float32)
    block_size = max(2, int(cfg.get("block_size", 3)))
    ksize = max(3, int(cfg.get("ksize", 3)))
    if ksize % 2 == 0:
        ksize += 1
    response = cv2.cornerMinEigenVal(gray, blockSize=block_size, ksize=ksize)
    response = np.maximum(response, 0)
    if response.max() > 0:
        response = response / response.max()
    return response.astype(np.float32)
