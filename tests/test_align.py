import pandas as pd
import numpy as np
import pytest
from clusterrender.transform.align import align_clusters

"""
Test the functions in the align module.
"""
# Make up test data
points_1 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1.1],
    }
)

# a little different than points_1
points_2 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1.11],
    }
)

# permuted points
points_1_perm = points_1.iloc[[1, 0, 2]].reset_index(drop=True)
points_1_perm2 = points_1.iloc[[0, 2, 1]].reset_index(drop=True)
points_2_perm = points_2.iloc[[1, 0, 2]].reset_index(drop=True)

# rotated points
theta = np.pi / 2  # 90 degrees
Rx = np.array(
    [
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ]
)
t = np.array([0, 0.05, 0])  # translation vector


def apply_transformation(points, R, t):
    """Apply rotation and translation to the points."""
    points_rotated = points.copy()
    points_rotated[["x", "y", "z"]] = points[["x", "y", "z"]].dot(R.T) + t
    return points_rotated


reflect = np.diag([-1, -1, -1])  # inversion = reflection through the origin


def apply_reflection(points, reflect):
    """Apply reflection to the points."""
    points_reflected = points.copy()
    points_reflected[["x", "y", "z"]] = points[["x", "y", "z"]].dot(reflect.T)
    return points_reflected


points_1_rotated = apply_transformation(points_1, Rx, 0)
points_2_rotated = apply_transformation(points_2, Rx, 0)

rotate_data = [
    # (points, gt_points, need_permute, allow_reflection, expected_output)
    # test 1: no transformation -> return the same points
    (points_1, points_1, False, False, points_1),
    # test 2: permute only
    (points_1_perm, points_1, True, False, points_1),
    # test 3: rotation only (no reflection)
    (points_1_rotated, points_1, False, False, points_1),
    # test 4: align points2 with points 1 with rotation
    (points_2_rotated, points_1, False, False, points_2),
    # test 5: with inversion
    (
        apply_reflection(points_1_rotated, reflect),
        points_1,
        False,
        True,
        points_1,
    ),
    # test 6: align points2 with points 1 with rotation, inversion,
    # and permutation
    (
        apply_reflection(points_2_rotated.iloc[[1, 0, 2]], reflect),
        points_1,
        True,
        True,
        points_2,
    ),
]


@pytest.mark.parametrize(
    "input, gt_points, need_permute, allow_reflection, expected_output",
    rotate_data,
)
def test_align_structures(
    input, gt_points, need_permute, allow_reflection, expected_output
):
    """Test the align_structures function."""
    result, _, _ = align_clusters(
        input, gt_points, need_permute, allow_reflection
    )
    # check if the aligned points are equal to the expected output
    result_coords = result[["x", "y", "z"]].values
    expected_output_coords = expected_output[["x", "y", "z"]].values
    # check if the aligned points are equal to the expected output
    # with some tolerance
    print(result_coords)
    print(expected_output_coords)
    assert (abs(result_coords - expected_output_coords) < 1e-8).all()
