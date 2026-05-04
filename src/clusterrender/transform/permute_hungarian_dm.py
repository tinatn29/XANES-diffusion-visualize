import numpy as np
from scipy.optimize import linear_sum_assignment
from clusterrender.transform.distance_matrix import distance_matrix

"""
Hungarian algorithm but based on a sorted list of distances.
- Compute a distance matrix.
- Each row of the distance matrix provides the distances
from one atom to all others.
- Sort every row to get a distance list for each atom.
- Compute a cost matrix where the cost of assigning atom i in cluster
to atom j in ref_cluster is the L2 distance between the distance lists
of atom i and atom j.
- Use the Hungarian algorithm to find the optimal assignment.
- Permute the cluster according to the optimal assignment.
"""


def _get_distance_lists(cluster):
    """Compute the distance lists for each atom in the cluster.

    Parameters
    ----------
    cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.

    Returns
    -------
    cluster_distance_lists : list of numpy.ndarray
        The list of each atom's list of distances (sorted)
        to all other atoms in `cluster`.
        Each array is sorted in ascending order of distance.
    """
    cluster_dm = distance_matrix(cluster)

    # Each row of the distance matrix gives us
    # the distances from one atom to all others
    # Sort every row to get a sorted distance list for each atom
    sorted_distance_lists = [
        sorted(cluster_dm[i]) for i in range(len(cluster))
    ]

    return sorted_distance_lists


def _get_cost_matrix(cluster, ref_cluster):
    """Compute the cost matrix for the Hungarian algorithm based on distance
    lists.

    Parameters
    ----------
    cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the reference cluster.
        Expected columns: 'x', 'y', 'z'.

    Returns
    -------
    cost_matrix : numpy.ndarray
        The 2D array where cost_matrix[i, j] is the cost of assigning
        atom i in `cluster` to atom j in `ref_cluster`.
        The cost is computed as the L2 distance between
        the sorted distance lists of the two atoms.
    """
    cluster_distance_lists = _get_distance_lists(cluster)
    ref_distance_lists = _get_distance_lists(ref_cluster)

    n = len(cluster)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Compute L2 distance between the distance lists
            # of atom i in cluster and atom j in ref_cluster
            cost_matrix[i, j] = np.linalg.norm(
                np.array(cluster_distance_lists[i])
                - np.array(ref_distance_lists[j])
            )

    return cost_matrix


def permute_cluster_hungarian(cluster, ref_cluster):
    """Permute the coordinates of a cluster to optimally match a reference
    cluster using the Hungarian algorithm based on distance matrix cost.

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
    cost_matrix = _get_cost_matrix(cluster, ref_cluster)
    # print(cost_matrix)

    # Solve the assignment problem using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Apply the optimal permutation to the cluster
    # col_indices gives us the optimal assignment:
    # atom col_indices[i] in cluster should match atom i in ref_cluster
    permuted_cluster = cluster.iloc[col_indices].reset_index(drop=True)

    return permuted_cluster


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    ref_cluster = pd.DataFrame(
        {"x": [0, 3, 0], "y": [0, 4, 0], "z": [1, 0, 0]}
    )

    pert_cluster = pd.DataFrame(
        {"x": [0, 3.1, 0], "y": [0, 4, 0], "z": [0.9, 0, 0]}
    )

    cluster = pert_cluster.iloc[[1, 0, 2]].reset_index(
        drop=True
    )  # Permuted version of ref_cluster

    permuted_cluster = permute_cluster_hungarian(cluster, ref_cluster)
    print("Original Cluster:")
    print(cluster)
    print("Reference Cluster:")
    print(ref_cluster)
    print("Permuted Cluster:")
    print(permuted_cluster)
