import numpy as np
import pandas as pd
from clusterrender.transform.permute import permute_cluster

"""
Kabsch algorithm to permute, rotate, and align a cluster
with a given reference cluster.
- Assume that the clusters are already centered around the origin.
- Two clusters are the same size.
- The cluster are in the same order in both clusters (after permutation).
"""


def _kabsch_align(
    P,
    Q,
    origin_P=np.array([0, 0, 0]),
    origin_Q=np.array([0, 0, 0]),
    allow_reflection=False,
):
    """Align point set P to point set Q using the Kabsch algorithm. (Q = ground
    truth, P = generated)

    Parameters:
        P: NxD numpy array of source cluster
        Q: NxD numpy array of target cluster

    Returns:
        R: Optimal rotation matrix
        t: Translation vector (from P to Q)
        P_aligned: Rotated + translated P
    """
    assert P.shape == Q.shape, "Point sets must have the same shape"

    # Compute covariance matrix
    H = P.T @ Q
    # Compute SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Check for reflection (determinant = -1)
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = origin_Q - R @ origin_P

    # Apply rotation and translation
    P_aligned = (R @ P.T).T + t

    return R, t, P_aligned


def align_clusters(
    cluster, ref_cluster, need_permute=True, allow_reflection=False
):
    """Align the generated cluster to the ground truth cluster using the Kabsch
    algorithm. The clusters are already centered around the origin.

    Parameters
    ----------
    cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the generated cluster.
        Expected columns: 'species', 'x', 'y', 'z'.
    ref_cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the reference cluster.
        Expected columns: 'species', 'x', 'y', 'z'.
    need_permute : bool, optional
        Whether to permute the cluster to match the ground truth.
        Default is True.
    allow_reflection : bool, optional
        Whether to allow reflection in the alignment. Default is False.

    Returns
    -------
    aligned_cluster : pandas.DataFrame
        The aligned cluster as a DataFrame with columns 'x', 'y', 'z'.
    R : numpy.ndarray
        The rotation matrix used for alignment.
    t : numpy.ndarray
        The translation vector used for alignment.
    """
    if need_permute:
        # find the best permutation compared to the ground truth
        cluster = permute_cluster(cluster, ref_cluster)

    R, t, cluster_aligned = _kabsch_align(
        cluster[["x", "y", "z"]].values,
        ref_cluster[["x", "y", "z"]].values,
        allow_reflection=allow_reflection,
    )
    # format the aligned cluster the same way
    cluster_aligned = pd.DataFrame(cluster_aligned, columns=["x", "y", "z"])
    # copy over all other columns (e.g. species) from the original cluster
    other_columns = cluster.drop(columns=["x", "y", "z"])
    # concatenate the other columns with the aligned coordinates
    cluster_aligned = pd.concat(
        [other_columns.reset_index(drop=True), cluster_aligned], axis=1
    )
    cluster_aligned.columns = cluster.columns  # ensure same column order
    return cluster_aligned, R, t
