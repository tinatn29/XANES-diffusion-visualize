import pandas as pd
import numpy as np
import pytest
from clusterrender.transform.distance_matrix import distance_matrix

"""
Test the distance_matrix function in transform module.
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

# set up input and output for tests
# for each case: (input1, input2, expected_output)
distance_matrix_data = [(points_1, dm_1), (points_2, dm_2)]


@pytest.mark.parametrize("input, expected_output", distance_matrix_data)
def test_distance_matrix(input, expected_output):
    """Test the distance_matrix function."""
    result = distance_matrix(input)
    # check if the calculated distance matrix is equal to the expected output
    # with some tolerance (since we're using euclidean distance)
    assert (abs(result - expected_output) < 1e-15).all()
