import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA

"""
Functions to find the "principal axis" of a local structure
and the rotation matrix to align the local structure with the principal axis.
"""


def get_rotation_matrix(v1, v2):
    """Get the rotation matrix to align vector v1 with vector v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the cross product and sine of the angle
    cross = np.cross(v1, v2)
    sin_theta = np.linalg.norm(cross)

    # Calculate the cosine of the angle
    cos_theta = np.dot(v1, v2)

    # Check if the vectors are parallel or anti-parallel
    if np.abs(sin_theta) < 1e-10:
        if np.abs(cos_theta - 1) < 1e-10:
            return np.eye(3)  # parallel: return identity matrix
        else:
            return np.eye(3) * -1  # anti-parallel: return -identity matrix

    # Create the skew-symmetric cross product matrix
    K = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ]
    )

    # Calculate the rotation matrix using Rodrigues' rotation formula
    R = np.eye(3) + K + K @ K * (1 - cos_theta) / (sin_theta**2)

    return R


def _sort_cluster_by_distance(cluster):
    """Sort the cluster by distance from the origin.

    The first row will be the atom closest to the origin (assumed to be
    the center).
    """
    if "distance" not in cluster.columns:
        cluster["distance"] = np.sqrt(
            cluster["x"] ** 2 + cluster["y"] ** 2 + cluster["z"] ** 2
        )
    cluster_sorted = cluster.sort_values(by="distance").reset_index(drop=True)
    return cluster_sorted


"""
Tetrahedron: principal axis is the vector pointing from the central atom
to one of the neighbors.
"""


def get_principal_axis_tetrahedron(cluster):
    """Get the principal axis of a tetrahedron. The principal axis is the
    vector from the center to one of the neighbors. Here we pick the neighbor
    i, where the vector (xi, yi, zi) is closest to the z-axis.

    This assumes the cluster center is the atom closest to the origin.

    Parameters
    ----------
    cluster: DataFrame
        The DataFrame containing the coordinates of the tetrahedron.

    Returns
    -------
    principal_axis: ndarray
        The principal axis vector.
    """
    # sort cluster first so the first row is the center atom
    # (closest to origin)
    sorted_cluster = _sort_cluster_by_distance(cluster)
    # Calculate the vector from the metal center to each neighbor
    vectors = sorted_cluster[["x", "y", "z"]].values
    # Find the neighbor with the vector closest to the z-axis
    # (i.e. the one with the largest z-coordinate)
    principal_axis = vectors[np.argmax(np.abs(vectors[:, 2]))]

    return principal_axis


"""
Functions to orient trigonal bipyramidal structures.
This relies on finding a plane that contains metal + 3 neighbors.
Principal axis is the normal vector of the plane.
"""


def _plane_combinations(sorted_cluster, num_plane_corners=3):
    """Get a list of combinations of indices containing the central atom and
    num_plane_corners neighbors.

    Parameters
    ----------
    sorted_cluster: DataFrame
        DataFrame containing the coordinates of the structure,
        already sorted by distance from the origin.
        The first row is the central atom (closest to origin).
        The remaining rows are the neighbors sorted by distance.
    num_plane_corners: int
        The number of neighbors to consider for combinations.

    Returns
    -------
    indices_list: list
        List of lists, each containing indices of the central atom
        and its neighbors.
    """
    # Get the metal index and neighbor indices
    n_atoms = len(sorted_cluster)
    neighbor_indices = list(range(1, n_atoms))
    # Get all combinations of num_plane_corners neighbors
    combinations_list = list(combinations(neighbor_indices, num_plane_corners))
    indices_list = []
    # each entry in indices_list looks like [0, n1, n2, n3]
    # where 0 is the index of the central atom
    # and n1, n2, n3 are the indices of the neighbors in the combination
    for comb in combinations_list:
        indices_list.append([0] + list(comb))
    return indices_list


def _pca_fit_plane(plane_indices, sorted_cluster):
    """Fit a plane to the cluster defined by the indices in the row."""
    vectors = sorted_cluster[["x", "y", "z"]].values
    plane_cluster = np.array([vectors[i] for i in plane_indices])
    centered = plane_cluster - np.mean(plane_cluster, axis=0)
    pca = PCA(n_components=3)
    pca.fit(centered)
    # normal vector (3rd component of PCA)
    normal_vector = pca.components_[2]
    # explained variance by 3rd component
    explained_variance = pca.explained_variance_ratio_[2]
    return normal_vector, explained_variance


def get_principal_axis_find_plane(cluster, num_plane_corners=3):
    """Get the principal axis by fitting a plane. The principal axis is the
    normal vector of the plane defined by the metal center and its 3 or 4
    neighbors (forming a trigonal or square base). We find the plane by fitting
    PCA to all combinations of plane indices. This function returns the normal
    vector of the plane with the smallest explained variance by the 3rd
    component (most planar).

    Parameters
    ----------
    cluster: DataFrame
        DataFrame containing the coordinates of the structure.
    num_plane_corners: int
        The number of neighbors (not counting the central atom)
        to consider for plane fitting.
        Default is 3, which is appropriate for trigonal bipyramidal structures.
        For square planar structures, set this to 4.

    Returns
    -------
    principal_axis: ndarray
        The principal axis vector.
    """
    # start with a sorted cluster so the first row is the center atom
    # (closest to origin)
    sorted_cluster = _sort_cluster_by_distance(cluster)
    indices_list = _plane_combinations(
        sorted_cluster, num_plane_corners=num_plane_corners
    )
    best_axis = None
    best_variance = 1

    for indices in indices_list:
        normal_vector, explained_variance = _pca_fit_plane(
            indices, sorted_cluster
        )
        if explained_variance < best_variance:
            best_variance = explained_variance
            best_axis = normal_vector

    if best_axis[2] < 0:
        # Ensure the z-component is positive
        best_axis = -best_axis

    return best_axis


"""
We can also orient the structure by aligning one of the "base" atoms
with the x-axis.
"""


def get_xy_vector(sorted_cluster):
    """Get the xy vector (z=0) pointing from the z-axis to the base atom with
    the largest x value."""
    # start with a sorted cluster so the first row is the center atom
    # (closest to origin)
    # base atoms are all atoms except the center, so we take sorted_cluster[1:]
    df_base = sorted_cluster[1:].copy()
    df_base["abs_z"] = np.abs(df_base["z"])
    # remove the two atoms with the largest absolute z values
    df_base = (
        df_base.sort_values(by="abs_z", ascending=True)
        .reset_index(drop=True)
        .tail(-2)
    )
    # choose the atom with the largest x value
    df_base = (
        df_base.sort_values(by="x", ascending=False)
        .reset_index(drop=True)
        .head(1)
    )
    xy_vector = np.array([df_base["x"].values[0], df_base["y"].values[0], 0])

    return xy_vector
