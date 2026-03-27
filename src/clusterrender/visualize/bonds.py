import numpy as np

"""Draw bonds as lines between atoms with specified colors and widths."""


def get_bond_list(
    df,
    bond_type="nearest_lower_shell",
    distance_cutoff=None,
):
    """Extract bonds from ClusterDataFrame using various criteria.

    Parameters
    ----------
    df : ClusterDataFrame
        The DataFrame containing the cluster of atoms,
        and optionally 'nearest_lower_shell_neighbor' and 'site_name'
        for shell-based bonding.
    bond_type : {"nearest_lower_shell", "distance_cutoff", "center_to_all"},
    default "nearest_lower_shell"
        The type of bonding criterion to use:
        - "nearest_lower_shell" : Connect atoms to their nearest
                                lower-shell neighbors using the
                                'nearest_lower_shell_neighbor' column
        - "distance_cutoff" : Connect atoms within a specified distance limit
        - "center_to_all" : Connect the center atom to all other atoms
    distance_cutoff : float, optional
        The cutoff distance in Å when bond_type="distance_cutoff".
        Required when using distance-based bonding.

    Returns
    -------
    bonds : list of tuple
        List of bond pairs as tuples (i, j) where i and j are DataFrame
        indices of bonded atoms. Each bond is represented once.

    Raises
    ------
    KeyError
        If required columns ('nearest_lower_shell_neighbor', 'site_name')
        are missing for shell-based bonding.
    ValueError
        If distance_cutoff is None when bond_type="distance_cutoff".

    Examples
    --------
    >>> # Shell-based bonding (default)
    >>> bonds = get_bond_list(cluster_df)

    >>> # Distance-based bonding
    >>> bonds = get_bond_list(cluster_df, bond_type="distance_cutoff",
    ...                       distance_cutoff=2.5)

    >>> # Center-to-all bonding
    >>> bonds = get_bond_list(cluster_df, bond_type="center_to_all")
    >>> print(bonds)
    [(0, 1), (0, 2), (0, 3), (0, 4)]

    Notes
    -----
    For shell-based bonding, the function uses the
    'nearest_lower_shell_neighbor' column in ClusterDataFrame.
    Distance-based bonding uses Euclidean distance in Å as cutoff.
    """
    bonds = []

    if bond_type == "nearest_lower_shell":
        # Use existing shell neighbor relationships
        for idx, row in df.iterrows():
            neighbor = row.get("nearest_lower_shell_neighbor")
            if neighbor and neighbor != "center":
                neighbor_idx = df[df["site_name"] == neighbor].index[0]
                bonds.append((neighbor_idx, idx))

    elif bond_type == "distance_cutoff":
        # Distance-based bonding
        coords = df[["x", "y", "z"]].values
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance <= distance_cutoff:
                    bonds.append((i, j))

    elif bond_type == "center_to_all":
        # Connect center atom to all others
        bonds = [(0, i) for i in range(1, len(df))]

    return bonds


def draw_bonds(
    df,
    ax,
    bonds=None,
    x_column="e1",
    y_column="e2",
    color="gray",
    width=6,
    alpha=0.8,
    style="-",
):
    """Draw bonds as lines between specified atom pairs."""
    # Get a list of bonds if not provided
    if bonds is None:
        bonds = get_bond_list(df)

    # Get coordinates
    x_coords = df[x_column].values
    y_coords = df[y_column].values

    # Draw all bonds efficiently using LineCollection for better performance
    from matplotlib.collections import LineCollection

    lines = []
    for i, j in bonds:
        lines.append([(x_coords[i], y_coords[i]), (x_coords[j], y_coords[j])])

    lc = LineCollection(
        lines,
        colors=color,
        linewidths=width,
        alpha=alpha,
        linestyles=style,
        zorder=1,
    )
    ax.add_collection(lc)
    return
