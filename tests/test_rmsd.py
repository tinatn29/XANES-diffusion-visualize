import pytest
import pandas as pd
import numpy as np
from clusterrender.compute.rmsd import root_mean_square_distance

"""
Test the mde calculator
"""
# test data
points_1 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1],
    }
)

points_2 = pd.DataFrame(
    {
        "species": ["M", "O", "O"],
        "x": [0, 0, 0],
        "y": [0, 0, 0],
        "z": [0, 1, -1.3],
    }
)

datasets = [
    # (cluster 1, cluster 2, expected RMSD)
    (points_1, points_1, 0.0),
    (points_1, points_2, np.sqrt(0.3**2 / 3)),
    (points_2, points_1, np.sqrt(0.3**2 / 3)),
]


@pytest.mark.parametrize("df, df_ref, expected_output", datasets)
def test_root_mean_square_distance(df, df_ref, expected_output):
    """Test the root_mean_square_distance function."""
    rmsd = root_mean_square_distance(df, df_ref)
    # assert up to a rounding error
    assert np.abs(rmsd - expected_output) < 1e-6
