import numpy as np

"""
Compute root mean square distance (RMSD) between two (transformed) clusters
The atoms in both clusters must be in the same order.
"""


def root_mean_square_distance(df, df_ref):
    """Compute root mean square distance across corresponding pairs of atoms in
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
        The root mean square distance between the two clusters. This is the
        square root of the average of the squared Euclidean distances between
        corresponding atom pairs.
    """
    # compute euclidean distances between corresponding pairs
    df_coords = df[["x", "y", "z"]].values
    df_ref_coords = df_ref[["x", "y", "z"]].values
    distances = np.linalg.norm(df_coords - df_ref_coords, axis=1)

    # compute root mean square distance
    rmsd = np.sqrt(np.mean(distances**2))

    return rmsd
