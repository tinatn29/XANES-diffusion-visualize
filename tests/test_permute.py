import pandas as pd
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

# load cdf from files
gt_cdf_1 = pd.read_pickle("tests/test_data/gt_1_output.pkl")
gt_cdf_1_full = pd.read_pickle("tests/test_data/test_groundtruth_output_1.pkl")
# permuted cdf_1
gt_cdf_1_perm = gt_cdf_1.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
gt_cdf_1_full_perm = gt_cdf_1_full.iloc[[2, 1, 0, 3, 4]].reset_index(drop=True)

permute_data = [
    # (points, gt_points, expected_output)
    (points_1_perm, points_1, points_1),
    (points_1, points_1, points_1),
    (points_1_perm2, points_1, points_1),
    # slightly different points, permuted
    (points_2_perm, points_1, points_2),
    # test with cdf data
    (gt_cdf_1, gt_cdf_1, gt_cdf_1),
    (gt_cdf_1_perm, gt_cdf_1, gt_cdf_1),
    (gt_cdf_1_full_perm, gt_cdf_1_full, gt_cdf_1_full),
]


@pytest.mark.parametrize("cluster, ref_cluster, expected_output", permute_data)
def test_permute_cluster(cluster, ref_cluster, expected_output):
    """Test the permute_cluster function."""
    permuted_cluster = permute_cluster(cluster, ref_cluster)
    print(permuted_cluster)
    print(expected_output)
    # check if the permuted points are equal to the expected output
    pd.testing.assert_frame_equal(
        permuted_cluster, expected_output, check_dtype=False
    )
