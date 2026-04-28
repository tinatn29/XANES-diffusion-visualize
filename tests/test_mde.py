import pytest
import pandas as pd
import numpy as np
from clusterrender.compute.mde import mean_distance_error

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
    # (cluster 1, cluster 2, expected MDE)
    (points_1, points_1, 0.0),
    (points_1, points_2, 0.1),
    (points_2, points_1, 0.1),
]


@pytest.mark.parametrize("df, df_ref, expected_output", datasets)
def test_mean_distance_error(df, df_ref, expected_output):
    """Test the mean_distance_error function."""
    mde = mean_distance_error(df, df_ref)
    # assert up to a rounding error
    assert np.abs(mde - expected_output) < 1e-6
