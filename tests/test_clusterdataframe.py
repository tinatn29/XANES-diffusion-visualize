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


# set up all test data from files
def read_pickle_and_map_columns(file_path):
    df = pd.read_pickle(file_path)
    df.columns = df.columns.map(str)
    return df


data = {
    # groundtruth rows as inputs
    "gt_row_1": pd.read_pickle("tests/test_data/test_groundtruth_row_1.pkl"),
    "gt_row_2": pd.read_pickle("tests/test_data/test_groundtruth_row_2.pkl"),
    # dataframe outputs from ground truth rows
    "df_gt_full_1": read_pickle_and_map_columns(
        "tests/test_data/test_groundtruth_output_1.pkl"
    ),
    "df_gt_full_2": read_pickle_and_map_columns(
        "tests/test_data/test_groundtruth_output_2.pkl"
    ),
    # xyz and txt files as inputs
    "gt_file_1": "tests/test_data/gt_1.xyz",
    "gt_file_1_txt": "tests/test_data/gt_1.txt",
    "gt_file_2": "tests/test_data/gt_2.xyz",
    "gen_file_1": "tests/test_data/gen_1.xyz",
    "gen_file_1_txt": "tests/test_data/gen_1.txt",
    "gen_file_2": "tests/test_data/gen_2.xyz",
    # dataframe outputs from xyz and txt files
    "df_gt_1": read_pickle_and_map_columns("tests/test_data/gt_1_output.pkl"),
    "df_gt_2": read_pickle_and_map_columns("tests/test_data/gt_2_output.pkl"),
    "df_gen_1": read_pickle_and_map_columns(
        "tests/test_data/gen_1_output.pkl"
    ),
    "df_gen_2": read_pickle_and_map_columns(
        "tests/test_data/gen_2_output.pkl"
    ),
    # dataframe outputs (centered)
    "df_centered_gen_1_Fe": read_pickle_and_map_columns(
        "tests/test_data/gen_1_centered_0.pkl"
    ),
    "df_centered_gen_1_O": read_pickle_and_map_columns(
        "tests/test_data/gen_1_centered_1.pkl"
    ),
}

test_gt_inputs = [
    (
        data["gt_row_1"],
        "Fe",
        data["df_gt_full_1"].drop(columns=["nearest_lower_shell_neighbor"]),
    ),
    (
        data["gt_row_2"],
        "Fe",
        data["df_gt_full_2"].drop(columns=["nearest_lower_shell_neighbor"]),
    ),
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
    (
        data["df_gt_full_1"].drop(columns=["nearest_lower_shell_neighbor"]),
        data["df_gt_full_1"],
    ),
    (
        data["df_gt_full_2"].drop(columns=["nearest_lower_shell_neighbor"]),
        data["df_gt_full_2"],
    ),
    (
        data["df_gt_full_1"].drop(
            columns=["nearest_lower_shell_neighbor", "site_name"]
        ),
        data["df_gt_full_1"],
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
cols_to_drop = ["nearest_lower_shell_neighbor", "site_name", "shell"]
test_xyz_inputs = [
    (data["gt_file_1"], data["df_gt_1"], None, None),
    (data["gt_file_1_txt"], data["df_gt_1"], None, None),
    (data["gen_file_1"], data["df_gen_1"], None, None),
    (data["gen_file_2"], data["df_gen_2"], None, None),
    (
        data["gt_file_1"],
        data["df_gt_full_1"].drop(columns=cols_to_drop),
        "Fe",
        None,
    ),
    (data["gen_file_1"], data["df_centered_gen_1_Fe"], None, 0),
]


@pytest.mark.parametrize(
    "file_path, expected_df, species, index", test_xyz_inputs
)
def test_clusterdataframe_from_xyz(file_path, expected_df, species, index):
    cdf = ClusterDataFrame.from_xyz(
        file_path, center_species=species, center_index=index
    )
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, expected_df, check_dtype=False)


# set up test data for center_cluster
cols_to_drop = ["nearest_lower_shell_neighbor", "site_name", "shell"]
test_center_inputs = [
    (
        data["df_gt_1"],
        "Fe",
        None,
        data["df_gt_full_1"].drop(columns=cols_to_drop),
    ),
    (
        data["df_gt_2"],
        "Fe",
        None,
        data["df_gt_full_2"].drop(columns=cols_to_drop),
    ),
    (
        data["df_gt_1"],
        None,
        0,
        data["df_gt_full_1"].drop(columns=cols_to_drop),
    ),
    (
        data["df_gt_2"],
        None,
        0,
        data["df_gt_full_2"].drop(columns=cols_to_drop),
    ),
    (data["df_gen_1"], "Fe", None, data["df_centered_gen_1_Fe"]),
    (data["df_gen_1"], None, 1, data["df_centered_gen_1_O"]),
]


@pytest.mark.parametrize(
    "input_df, species, index, expected_df", test_center_inputs
)
def test_clusterdataframe_center_cluster(
    input_df, species, index, expected_df
):
    cdf = ClusterDataFrame(input_df)
    cdf.center_cluster(center_species=species, center_index=index)
    print(cdf)
    print(expected_df)
    # check type and content
    assert isinstance(cdf, ClusterDataFrame)
    pd.testing.assert_frame_equal(cdf, expected_df, check_dtype=False)
