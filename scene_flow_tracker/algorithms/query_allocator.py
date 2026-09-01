from __future__ import annotations


def allocate_initial_queries(left_valid: bool, right_valid: bool, total_query_points: int, points_per_detected_arm: int) -> dict[str, int]:
    total = int(total_query_points)
    per_arm = int(points_per_detected_arm)
    if total <= 0:
        raise ValueError("total_query_points must be > 0")
    if per_arm < 0:
        raise ValueError("points_per_detected_arm must be >= 0")
    if 2 * per_arm > total:
        raise ValueError("2 * points_per_detected_arm must be <= total_query_points")
    left = per_arm if left_valid else 0
    right = per_arm if right_valid else 0
    return {"left": left, "right": right, "environment": total - left - right}


def final_environment_target(total_query_points: int, actual_left_count: int, actual_right_count: int) -> int:
    target = int(total_query_points) - int(actual_left_count) - int(actual_right_count)
    if target < 0:
        raise ValueError("robot sample count exceeds total_query_points")
    return target
