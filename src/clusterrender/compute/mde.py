import numpy as np

"""
Compute MDE between two (transformed) clusters
The atoms in both clusters must be in the same order.
"""


def mean_distance_error(df, df_ref):
    """Compute mean euclidean distance across corresponding pairs of atoms in
    two clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing atomic coordinates for the first cluster.
        Must contain columns 'x', 'y', 'z' with atomic positions.
    df_ref : pandas.DataFrame
        The DataFrame containing atomic coordinates for the reference cluster.
        Must contain columns 'x', 'y', 'z' with atomic positions.
        Must have the same number of atoms as `df` and atoms must be
        in corresponding order.

    Returns
    -------
    float
        The mean distance error between the two clusters. This is the
        average of the Euclidean distances between corresponding atom pairs.

    Notes
    -----
    The atoms in both clusters must be in the same order for meaningful
    comparison. No alignment or reordering is performed by this function.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'x': [0, 1], 'y': [0, 0], 'z': [0, 0]})
    >>> df2 = pd.DataFrame({'x': [0, 2], 'y': [0, 0], 'z': [0, 0]})
    >>> mean_distance_error(df1, df2)
    0.5
    """
    # compute euclidean distances between corresponding pairs
    df_coords = df[["x", "y", "z"]].values
    df_ref_coords = df_ref[["x", "y", "z"]].values
    distances = np.linalg.norm(df_coords - df_ref_coords, axis=1)

    # compute mean distance error
    mde = np.mean(distances)

    return mde
