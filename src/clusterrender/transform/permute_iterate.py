from clusterrender.transform.permute_hungarian import permute_hungarian
from clusterrender.compute.mde import mean_distance_error

"""
Iteratively permute and align a cluster to best match a reference cluster.
"""


def permute_iterate(cluster, ref_cluster, max_iterations=10):
    """Iteratively permute and align a cluster to best match a reference
    cluster.

    Parameters
    ----------
    cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.
    ref_cluster : pandas.DataFrame
        The DataFrame containing the coordinates of the reference cluster.
        Expected columns: 'x', 'y', 'z'.
    max_iterations : int, optional
        Maximum number of iterations to perform. Default is 10.

    Returns
    -------
    permuted_cluster : pandas.DataFrame
        The permuted cluster that best matches the reference cluster.
    """
    permuted_cluster = cluster.copy()

    min_mde = mean_distance_error(permuted_cluster, ref_cluster)
    optimal_perm = cluster.copy()

    print("Initial MDE:", min_mde)

    for iteration in range(max_iterations):
        # Permute the cluster to best match the reference cluster
        permuted_cluster = permute_hungarian(permuted_cluster, ref_cluster)

        current_mde = mean_distance_error(permuted_cluster, ref_cluster)
        print(f"Iteration {iteration + 1} MDE:", current_mde)
        if current_mde < min_mde:
            min_mde = current_mde
            optimal_perm = permuted_cluster.copy()

    print("Min MDE:", min_mde)

    return optimal_perm
