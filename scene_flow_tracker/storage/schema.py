from __future__ import annotations

from enum import IntEnum


SCHEMA_VERSION = "1.2"
FRAME_RANGE_SEMANTICS = "half_open"
TRACK_COORDINATE_ORDER = "xy"
TRACK_COORDINATE_SPACE = "original_image_pixels"
YOLO_CLASS_NAMES = ("left_arm", "right_arm")
GROUPS = ("left", "right", "env")
GROUP_TO_RESULT_KEY = {"left": "left", "right": "right", "env": "environment"}


class SegmentStatus(IntEnum):
    UNKNOWN_OR_PADDING = 0
    OK = 1
    PARTIAL = 2
    FAILED = 3


class GroupStatus(IntEnum):
    UNKNOWN = 0
    OK = 1
    NO_DETECTION = 2
    NO_CANDIDATES = 3
    TRACK_FAILED = 4
    PROCESSING_FAILED = 5


class TrackState(IntEnum):
    INVALID_OR_PADDING = 0
    VALID = 1
    PARTIAL = 2
    FAILED = 3


class MotionState(IntEnum):
    UNKNOWN_OR_PADDING = 0
    STATIC = 1
    MOVING = 2
    JITTER = 3
    UNCERTAIN = 4


class CandidateLevel(IntEnum):
    INVALID_OR_PADDING = 0
    ROBOT_CANDIDATE = 1
    AMBIGUOUS = 2
    BACKGROUND_CANDIDATE = 3


TRACK_STATE_MAP = {
    "valid": TrackState.VALID,
    "partial": TrackState.PARTIAL,
    "failed": TrackState.FAILED,
    "invalid": TrackState.INVALID_OR_PADDING,
    "padding": TrackState.INVALID_OR_PADDING,
    "unknown": TrackState.INVALID_OR_PADDING,
}

MOTION_STATE_MAP = {
    "static": MotionState.STATIC,
    "moving": MotionState.MOVING,
    "jitter": MotionState.JITTER,
    "uncertain": MotionState.UNCERTAIN,
    "unknown": MotionState.UNKNOWN_OR_PADDING,
    "padding": MotionState.UNKNOWN_OR_PADDING,
}

CANDIDATE_LEVEL_MAP = {
    "robot_candidate": CandidateLevel.ROBOT_CANDIDATE,
    "robot": CandidateLevel.ROBOT_CANDIDATE,
    "ambiguous": CandidateLevel.AMBIGUOUS,
    "background_candidate": CandidateLevel.BACKGROUND_CANDIDATE,
    "background": CandidateLevel.BACKGROUND_CANDIDATE,
    "invalid": CandidateLevel.INVALID_OR_PADDING,
    "padding": CandidateLevel.INVALID_OR_PADDING,
    "unknown": CandidateLevel.INVALID_OR_PADDING,
}

ENUM_METADATA = {
    "segment_status": {int(v): v.name for v in SegmentStatus},
    "group_status": {int(v): v.name for v in GroupStatus},
    "track_state": {int(v): v.name for v in TrackState},
    "motion_state": {int(v): v.name for v in MotionState},
    "candidate_level": {int(v): v.name for v in CandidateLevel},
    "yolo_classes": {i: name for i, name in enumerate(YOLO_CLASS_NAMES)},
}

FILTER_FEATURE_ALIASES = {
    "visibility_ratio": ("visibility_ratio",),
    "net_displacement": ("net_displacement", "net_displacement_px"),
    "path_length": ("path_length", "path_length_px"),
    "path_efficiency": ("path_efficiency",),
    "jitter_rms": ("jitter_rms", "jitter_rms_px"),
    "jitter_residual_ratio": ("jitter_residual_ratio",),
    "turn_consistency": ("turn_consistency",),
    "turn_angle_mad": ("turn_angle_mad",),
    "normalized_jerk": ("normalized_jerk",),
    "direction_reversal_ratio": ("direction_reversal_ratio",),
}

ROBOT_SAMPLING_FEATURES = (
    "sampling_score",
    "edge_strength",
    "trackability_score",
    "color_score",
    "topology_score",
)

ENV_SAMPLING_FEATURES = (
    "sampling_score",
    "edge_strength",
    "trackability_score",
)
