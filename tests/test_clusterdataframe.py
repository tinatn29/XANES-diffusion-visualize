import pandas as pd
import pytest
from clusterrender.clusterdataframe import ClusterDataFrame

"""Unit tests for ClusterDataFrame class."""
# set up test data for init
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


# set up test data for from_groundtruth_row
gt_row_1 = pd.read_pickle("tests/test_data/test_groundtruth_row_1.pkl")
gt_row_2 = pd.read_pickle("tests/test_data/test_groundtruth_row_2.pkl")
df_out_1 = pd.read_pickle("tests/test_data/test_groundtruth_output_1.pkl")
df_out_2 = pd.read_pickle("tests/test_data/test_groundtruth_output_2.pkl")

# map columns to string
df_out_1.columns = df_out_1.columns.map(str)
df_out_2.columns = df_out_2.columns.map(str)

test_gt_inputs = [
    (gt_row_1, "Fe", df_out_1.drop(columns=["nearest_lower_shell_neighbor"])),
    (gt_row_2, "Fe", df_out_2.drop(columns=["nearest_lower_shell_neighbor"])),
]


@pytest.mark.parametrize("gt_row, central_atom, expected_df", test_gt_inputs)
def test_clusterdataframe_from_groundtruth_row(
    gt_row, central_atom, expected_df
):
    cdf = ClusterDataFrame.from_groundtruth_row(gt_row, central_atom)
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, expected_df, check_dtype=False)


# set up test data for add_closest_lower_shell_neighbor
test_lower_shell_inputs = [
    (df_out_1.drop(columns=["nearest_lower_shell_neighbor"]), df_out_1),
    (df_out_2.drop(columns=["nearest_lower_shell_neighbor"]), df_out_2),
    (
        df_out_1.drop(columns=["nearest_lower_shell_neighbor", "site_name"]),
        df_out_1,
    ),
]


@pytest.mark.parametrize("input_df, expected_df", test_lower_shell_inputs)
def test_clusterdataframe_add_closest_lower_shell_neighbor(
    input_df, expected_df
):
    cdf = ClusterDataFrame(input_df)
    cdf.add_closest_lower_shell_neighbor()
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, expected_df, check_dtype=False)


# set up test data for from_xyz
output_gt_1 = pd.read_pickle("tests/test_data/gt_1_output.pkl")
output_gen_1 = pd.read_pickle("tests/test_data/gen_1_output.pkl")
output_gen_2 = pd.read_pickle("tests/test_data/gen_2_output.pkl")
# map columns to string
output_gt_1.columns = output_gt_1.columns.map(str)
output_gen_1.columns = output_gen_1.columns.map(str)
output_gen_2.columns = output_gen_2.columns.map(str)

test_xyz_inputs = [
    ("tests/test_data/gt_1.xyz", output_gt_1),
    ("tests/test_data/gt_1.txt", output_gt_1),
    ("tests/test_data/gen_1.xyz", output_gen_1),
    ("tests/test_data/gen_2.xyz", output_gen_2),
]


@pytest.mark.parametrize("file_path, expected_df", test_xyz_inputs)
def test_clusterdataframe_from_xyz(file_path, expected_df):
    cdf = ClusterDataFrame.from_xyz(file_path)
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, expected_df, check_dtype=False)
