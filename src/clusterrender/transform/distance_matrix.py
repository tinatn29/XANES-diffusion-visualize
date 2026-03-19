import numpy as np


def distance_matrix(cluster):
    """Compute the distance matrix for a set of points given in a DataFrame.
    Each entry (i, j) in the matrix is the distance between points i and j.

    Parameters
    ----------
    cluster : pandas.DataFrame
        A DataFrame containing the coordinates of the points.
        Expected columns: 'x', 'y', 'z'.

    Returns
    -------
    dist_matrix : numpy.ndarray
        A 2D array representing the distance matrix.
    """
    n = len(cluster)  # no. of points
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = np.linalg.norm(
                cluster.iloc[i][["x", "y", "z"]].values
                - cluster.iloc[j][["x", "y", "z"]].values
            )
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix
