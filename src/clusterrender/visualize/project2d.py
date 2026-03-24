import numpy as np

"""
Project 3D coordinates onto a 2D plane
"""


def project_to_plane(cluster, azimuthal_angle, tilt_angle):
    """Project 3D coordinates onto a 2D plane, defined by a normal vector,
    which is set by azimuthal and tilt angles.

    Parameters
    ----------
    cluster: pandas.DataFrame or ClusterDataFrame
        The DataFrame containing the coordinates of the cluster.
    azimuthal_angle: float
        The angle in degrees defining the normal vector in the XY plane.
    tilt_angle: float
        The angle in degrees defining the tilt of the normal vector
        from the Z axis.

    Returns
    -------
    projected_coords: numpy array
        The 2D coordinates of the projected points
        in the same order as the input DataFrame.
    """
    # Convert azimuthal angle to radians
    phi = np.radians(azimuthal_angle)

    # Set small tilt angle from vertical (in degrees)
    tilt_rad = np.radians(tilt_angle)
    epsilon = np.tan(tilt_rad)  # small z component

    # Normal vector: mostly in XY, slight z tilt
    n = np.array([np.cos(phi), np.sin(phi), epsilon])
    n = n / np.linalg.norm(n)  # normalize

    coords = cluster[["x", "y", "z"]].values
    proj_points = coords - np.dot(coords, n)[:, None] * n

    # Define an orthonormal 2D basis in the plane
    # e1: orthogonal to n and pointing in Z
    e1 = np.cross(n, [0, 0, 1])
    e1 /= np.linalg.norm(e1)
    # e2: Z-axis
    e2 = np.array([0, 0, 1])

    # Step 3: Express projected 3D points in this 2D basis
    projected_coords = np.stack(
        [np.dot(proj_points, e1), np.dot(proj_points, e2)], axis=1
    )

    return projected_coords
