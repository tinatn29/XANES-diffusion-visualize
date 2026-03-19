import numpy as np
from clusterrender.transform.distance_matrix import distance_matrix

"""
Helper functions to find the optimal permutation of the coordinates.
"""


def _distance_score(cluster, ref_cluster_dm):
    """Compute the distance score between two sets clusters based on the
    distance matrices. The two clusters must be the same size.

    Parameters
    ----------
    cluster : pandas.DataFrame
        A DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster_dm : numpy.ndarray
        A 2D array representing the distance matrix of the reference cluster.

    Returns
    -------
    distance_score : float
        L2 difference between the distance matrices of the cluster
        and the reference cluster.
    """
    dm = distance_matrix(cluster)
    # as a start - let's use euclidean distance between two distance matrices
    return np.linalg.norm(dm - ref_cluster_dm)


def _find_best_permutation(cluster, ref_cluster_dm):
    """Find the best permutation of the coordinates that minimizes the distance
    score.

    Parameters
    ----------
    cluster : pandas.DataFrame
        A DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster_dm : numpy.ndarray
        A 2D array representing the distance matrix of the reference cluster.

    Returns
    -------
    best_permutation : list
        The best permutation of the indices of the cluster.
    best_score : float
        The best distance score achieved by the best permutation.
    """
    from itertools import permutations

    n = len(cluster)
    indices = list(range(n))
    best_score = float("inf")
    best_permutation = None

    for perm in permutations(indices):
        permuted_cluster = cluster.iloc[list(perm)].reset_index(drop=True)
        score = _distance_score(permuted_cluster, ref_cluster_dm)
        if score < best_score:
            best_score = score
            best_permutation = perm

    return best_permutation, best_score


def permute_cluster(cluster, ref_cluster):
    """Permute the coordinates of the cluster to best match the reference
    cluster.

    Parameters
    ----------
    cluster : pandas.DataFrame
        A DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster : pandas.DataFrame
        A DataFrame containing the coordinates of the reference cluster.
        Expected columns: 'x', 'y', 'z'.

    Returns
    -------
    permuted_cluster : pandas.DataFrame
        The permuted cluster that best matches the reference cluster.
    """
    ref_cluster_dm = distance_matrix(ref_cluster)
    best_permutation, _ = _find_best_permutation(cluster, ref_cluster_dm)
    permuted_cluster = cluster.iloc[list(best_permutation)].reset_index(
        drop=True
    )
    return permuted_cluster
