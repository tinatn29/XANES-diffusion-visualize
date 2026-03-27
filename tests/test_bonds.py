import pytest
import pandas as pd
from clusterrender.clusterdataframe import ClusterDataFrame
from clusterrender.visualize.bonds import get_bond_list

"""Unit tests for bonds module."""


# Helper function to read pickle data
def read_pickle_and_map_columns(file_path):
    df = pd.read_pickle(file_path)
    df.columns = df.columns.map(str)
    return df


# Load test data
test_cdf_1 = read_pickle_and_map_columns(
    "tests/test_data/test_groundtruth_output_1.pkl"
)
test_cdf_2 = read_pickle_and_map_columns(
    "tests/test_data/test_groundtruth_output_2.pkl"
)

# Expected bonds for each dataset based on nearest_lower_shell_neighbor column
# Simple: all connect to center
expected_bonds_1 = [(0, 1), (0, 2), (0, 3), (0, 4)]
# More complex: 1st and 2nd shell
expected_bonds_2 = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (4, 5),
    (2, 6),
    (1, 7),
    (3, 8),
    (4, 9),
    (3, 10),
]

# Test data tuples: (test_data, expected_bonds)
shell_bonding_test_cases = [
    (test_cdf_1, expected_bonds_1),
    (test_cdf_2, expected_bonds_2),
]


@pytest.mark.parametrize("cdf, expected_bonds", shell_bonding_test_cases)
def test_nearest_lower_shell_bonding(cdf, expected_bonds):
    """Test bond extraction using nearest_lower_shell_neighbor column."""
    # Create ClusterDataFrame from test data
    cdf = ClusterDataFrame(cdf)

    # Get bonds
    bonds = get_bond_list(cdf, bond_type="nearest_lower_shell")

    # Sort bonds for comparison since order might vary
    bonds_sorted = sorted(bonds)
    expected_bonds_sorted = sorted(expected_bonds)

    assert bonds_sorted == expected_bonds_sorted
    assert len(bonds) == len(expected_bonds)


# Test data tuples: (test_data, expected_bonds)
test_coords = pd.DataFrame(
    {
        "species": ["A", "B", "C"],
        "x": [0.0, 1.0, 3.0],
        "y": [0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0],
        "site_name": ["A_0", "B_1", "C_2"],
    }
)

cutoff_bonding_test_cases = [
    (test_cdf_1, expected_bonds_1, 3.0),  # cutoff that includes all bonds
    (
        ClusterDataFrame(test_coords),
        [(0, 1)],
        1.5,
    ),  # cutoff that only includes A-B bond
]


@pytest.mark.parametrize(
    "cdf, expected_bonds, distance_cutoff", cutoff_bonding_test_cases
)
def test_distance_cutoff_bonding(cdf, expected_bonds, distance_cutoff):
    """Test bond extraction using distance cutoff."""
    cdf = ClusterDataFrame(cdf)
    bonds = get_bond_list(
        cdf, bond_type="distance_cutoff", distance_cutoff=distance_cutoff
    )
    assert bonds == expected_bonds


all_bonding_test_cases = [(test_cdf_1, expected_bonds_1)]  # includes all bonds


@pytest.mark.parametrize("cdf, expected_bonds", all_bonding_test_cases)
def test_center_to_all_bonding(cdf, expected_bonds):
    """Test bond extraction connecting center to all other atoms."""
    # Get bonds
    bonds = get_bond_list(cdf, bond_type="center_to_all")

    # Sort bonds for comparison
    bonds_sorted = sorted(bonds)
    expected_bonds_sorted = sorted(expected_bonds)

    assert bonds_sorted == expected_bonds_sorted
    assert len(bonds) == len(cdf) - 1
