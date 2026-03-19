import pandas as pd
import numpy as np
import pytest
from clusterrender.transform.permute import permute_cluster


# Make up test data
points_1 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1.1],
    }
)
dm_1 = np.array([[0, 1, 1.1], [1, 0, 2.1], [1.1, 2.1, 0]])

# a little different than points_1
points_2 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1.11],
    }
)
dm_2 = np.array([[0, 1, 1.11], [1, 0, 2.11], [1.11, 2.11, 0]])

# permuted points
points_1_perm = points_1.iloc[[1, 0, 2]].reset_index(drop=True)
points_1_perm2 = points_1.iloc[[0, 2, 1]].reset_index(drop=True)
points_2_perm = points_2.iloc[[1, 0, 2]].reset_index(drop=True)

permute_data = [
    # (points, gt_points, expected_output)
    (points_1_perm, points_1, points_1),
    (points_1, points_1, points_1),
    (points_1_perm2, points_1, points_1),
    # slightly different points, permuted
    (points_2_perm, points_1, points_2),
]


@pytest.mark.parametrize("input, gt_points, expected_output", permute_data)
def test_permute_cluster(input, gt_points, expected_output):
    """Test the permute_cluster function."""
    result = permute_cluster(input, gt_points)
    # check if the permuted points are equal to the expected output
    assert result.equals(expected_output)
