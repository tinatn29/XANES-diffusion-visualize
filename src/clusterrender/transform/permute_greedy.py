import numpy as np

"""
Permute the coordinates of a cluster to best match a reference cluster
using a greedy algorithm.
"""


def permute_greedy(cluster, ref_cluster):
    """Permute the coordinates of a cluster to best match the reference cluster
    using a greedy algorithm.

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
    permuted_cluster : pandas.DataFrame
        The permuted cluster that best matches the reference cluster.
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
            # Euclidean distance between atoms i and j
            cost_matrix[i, j] = np.linalg.norm(
                cluster_coords[i] - ref_coords[j]
            )

    # Greedy assignment: for each atom in cluster, find the closest atom
    # in ref_cluster that has not been assigned yet
    pairs = {}
    used_indices = set()
    used_ref_indices = set()

    cost = [
        (i, j, cost_matrix[i, j])
        for i in range(n_atoms)
        for j in range(n_atoms)
    ]
    cost.sort(key=lambda x: x[2])  # sort by cost (distance)

    for i, j, c in cost:
        if i not in used_indices and j not in used_ref_indices:
            pairs[i] = j
            used_indices.add(i)
            used_ref_indices.add(j)

    # Create permuted cluster based on the pairs
    permuted_cluster = cluster.copy()
    for i, j in pairs.items():
        permuted_cluster.iloc[i] = cluster.iloc[j]

    return permuted_cluster
