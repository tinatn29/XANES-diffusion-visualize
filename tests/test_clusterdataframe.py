import pandas as pd
import pytest
from clusterrender.clusterdataframe import ClusterDataFrame

"""Unit tests for ClusterDataFrame class."""
# set up test data
dict_input = {
    "species": ["C", "O", "O"],
    "x": [0.0, 1.0, -1.0],
    "y": [0.0, 0.0, 0.0],
    "z": [0.0, 0.0, 0.0],
}
df_input = pd.DataFrame(dict_input)

test_inputs = [
    dict_input,
    df_input,
]


@pytest.mark.parametrize("input_data", test_inputs)
def test_clusterdataframe_init(input_data):
    cdf = ClusterDataFrame(input_data)
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, pd.DataFrame(input_data))
