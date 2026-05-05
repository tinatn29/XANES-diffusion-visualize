import numpy as np
from scipy.optimize import linear_sum_assignment


def permute_hungarian(cluster, ref_cluster):
    """Permute the coordinates of a cluster to optimally match a reference
    cluster using the Hungarian algorithm.

    This function finds the optimal 1:1 assignment of atoms between two
    clusters by minimizing the sum of euclidean distances
    between matched atoms.
    The Hungarian algorithm guarantees finding the globally optimal solution.

    Parameters
    ----------
    cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the points to be permuted.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the reference cluster.
        Expected columns: 'x', 'y', 'z'.
        Must have the same number of atoms as `cluster`.

    Returns
    -------
    permuted_cluster : pandas.DataFrame
        The permuted cluster that optimally matches the reference cluster.
        Same structure as input `cluster` but with reordered rows.
    """
    # Extract coordinates as numpy arrays
    cluster_coords = cluster[["x", "y", "z"]].values
    ref_coords = ref_cluster[["x", "y", "z"]].values

    # Verify both clusters have same number of atoms
    if len(cluster_coords) != len(ref_coords):
        raise ValueError(
            f"Clusters must have the same number of atoms. "
            f"Got {len(cluster_coords)} and {len(ref_coords)} atoms."
        )

    # Create cost matrix: cost[i,j] = euclidean distance between
    # atom i in cluster and atom j in ref_cluster
    n_atoms = len(cluster_coords)
    cost_matrix = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            # Euclidean distance squared between atoms i and j
            cost_matrix[i, j] = (
                np.linalg.norm(cluster_coords[i] - ref_coords[j]) ** 2
            )

    # Solve the assignment problem using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Apply the optimal permutation to the cluster
    # col_indices gives us the optimal assignment:
    # atom row_indices[i] in cluster should match atom col_indices[i]
    # in ref_cluster
    permuted_cluster = cluster.iloc[col_indices].reset_index(drop=True)

    return permuted_cluster
