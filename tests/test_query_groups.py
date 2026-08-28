import numpy as np

from scene_flow_tracker.query_groups import GROUP_ID, merge_queries, split_by_layout


def test_query_groups_round_trip():
    left = np.array([[1, 2], [3, 4]], np.float32)
    right = np.array([[5, 6]], np.float32)
    env = np.array([[7, 8], [9, 10], [11, 12]], np.float32)
    merged, layout = merge_queries({"left": left, "right": right, "environment": env})
    assert merged.shape == (6, 2)
    assert layout.group_id.tolist() == [GROUP_ID["left"], GROUP_ID["left"], GROUP_ID["right"], GROUP_ID["environment"], GROUP_ID["environment"], GROUP_ID["environment"]]
    split = split_by_layout(merged, layout)
    np.testing.assert_array_equal(split["left"], left)
    np.testing.assert_array_equal(split["right"], right)
    np.testing.assert_array_equal(split["environment"], env)


def test_empty_group_is_two_dimensional():
    merged, layout = merge_queries({"left": np.empty((0, 2), np.float32), "right": np.ones((1, 2), np.float32)})
    assert merged.shape == (1, 2)
    assert layout.counts["left"] == 0
    assert split_by_layout(merged, layout)["left"].shape == (0, 2)
