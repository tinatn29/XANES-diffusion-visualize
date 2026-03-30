from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import numpy as np
from clusterrender.visualize.project2d import project_to_plane
from clusterrender.visualize.cluster import draw_cluster
from clusterrender.visualize.bonds import draw_bonds, get_bond_list
import matplotlib.pyplot as plt

"""
Main rendering function to visualize a cluster projected on a 2D plane.
This function integrates the projection of 3D coordinates, drawing of atoms,
and rendering of bonds between atoms.
"""


@dataclass
class ClusterStyle:
    """Styling parameters for atoms in cluster visualization.

    Attributes
    ----------
    override_colors : dict or array-like, optional
        Custom colors for atoms. Can be:
        - dict: {species: color} mapping (e.g., {'Fe': 'red', 'O': 'blue'})
        - array-like: Custom colors for each atom.
        Shape should be (n_atoms, 3) or (n_atoms,)
    override_radii : dict or array-like, optional
        Custom radii for atoms. Can be:
        - dict: {species: radius} mapping (e.g., {'Fe': 1.2, 'O': 0.8})
        - array-like: Custom radii for each atom. Shape should be (n_atoms,)
    """

    x_column: str = "e1"
    y_column: str = "e2"
    species_column: str = "species"
    scale: float = 2500
    alpha: float = 1.0
    override_colors: Optional[Union[Dict[str, Any], np.ndarray]] = None
    override_radii: Optional[Union[Dict[str, float], np.ndarray]] = None
    uniform_color: Optional[str] = None
    uniform_radius: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError("ClusterStyle alpha must be between 0 and 1")


@dataclass
class BondStyle:
    """Styling parameters for bonds in cluster visualization."""

    bond_type: str = "nearest_lower_shell"
    distance_cutoff: Optional[float] = None
    x_column: str = "e1"
    y_column: str = "e2"
    color: str = "gray"
    width: float = 6
    alpha: float = 0.8
    style: str = "-"

    def __post_init__(self):
        if self.bond_type not in [
            "nearest_lower_shell",
            "distance_cutoff",
            "center_to_all",
        ]:
            raise ValueError(
                "Invalid bond_type. Must be 'nearest_lower_shell', "
                "'distance_cutoff', or 'center_to_all'."
            )
        if (
            self.bond_type == "distance_cutoff"
            and self.distance_cutoff is None
        ):
            raise ValueError(
                "distance_cutoff required when bond_type='distance_cutoff'"
            )
        if not 0 <= self.alpha <= 1:
            raise ValueError("BondStyle alpha must be between 0 and 1")


def render_cluster(
    cluster,
    azimuthal_angle,
    tilt_angle,
    draw_bonds_flag=True,
    cluster_style: Optional[ClusterStyle] = None,
    bond_style: Optional[BondStyle] = None,
):
    """Render a cluster of atoms projected onto a 2D plane.

    Parameters
    ----------
    cluster : pandas.DataFrame or ClusterDataFrame
        The DataFrame containing the coordinates and species of the cluster.
    azimuthal_angle : float
        The angle in degrees defining the normal vector in the XY plane
        for projection.
    tilt_angle : float
        The angle in degrees defining the tilt of the normal vector
        from the Z axis for projection.
    draw_bonds_flag : bool, default True
        Whether to draw bonds between atoms.
    cluster_style : ClusterStyle, optional
        Styling parameters for atoms. If None, uses default ClusterStyle().
    bond_style : BondStyle, optional
        Styling parameters for bonds. If None, uses default BondStyle().

    Returns
    -------
    fig, ax : matplotlib figure and axes
        The rendered figure and axes with the projected cluster visualization.

    Examples
    --------
    >>> # Basic usage with defaults
    >>> fig, ax = render_cluster(my_cluster, 45, 30)

    >>> # Custom atom styling with uniform properties
    >>> custom_atoms = ClusterStyle(scale=300, alpha=0.7, uniform_color='red')
    >>> fig, ax = render_cluster(my_cluster, 45, 30,
                cluster_style=custom_atoms)

    >>> # Custom colors by species using dictionary
    >>> atom_colors = ClusterStyle(override_colors={'Fe': 'red',
                                                'O': 'blue', 'Ti': 'green'})
    >>> fig, ax = render_cluster(my_cluster, 45, 30, cluster_style=atom_colors)

    >>> # Custom radii by species
    >>> atom_radii = ClusterStyle(override_radii={'Fe': 1.2, 'O': 0.8})
    >>> fig, ax = render_cluster(my_cluster, 45, 30, cluster_style=atom_radii)

    >>> # Custom bond styling
    >>> custom_bonds = BondStyle(color='blue', width=4,
                            bond_type='distance_cutoff', distance_cutoff=2.5)
    >>> fig, ax = render_cluster(my_cluster, 45, 30, bond_style=custom_bonds)

    >>> # Both custom styles with species-specific colors
    >>> fig, ax = render_cluster(
    ...     my_cluster, 45, 30,
    ...     cluster_style=ClusterStyle(override_colors={'Fe': 'red',
    ...                                                'O': 'blue'}),
    ...     bond_style=BondStyle(color='gray', alpha=0.5)
    ... )
    """
    # Use defaults if not provided
    cluster_style = cluster_style or ClusterStyle()
    bond_style = bond_style or BondStyle()

    # Step 1: Project 3D coordinates to 2D plane
    projected_coords = project_to_plane(cluster, azimuthal_angle, tilt_angle)

    # Step 2: Create a new DataFrame with projected coordinates
    projected_df = cluster.copy()
    projected_df["e1"] = projected_coords[:, 0]
    projected_df["e2"] = projected_coords[:, 1]

    # Step 3: Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Step 4: Prepare parameters for drawing functions
    cluster_kwargs = {
        "x_column": cluster_style.x_column,
        "y_column": cluster_style.y_column,
        "species_column": cluster_style.species_column,
        "scale": cluster_style.scale,
        "alpha": cluster_style.alpha,
        "override_colors": cluster_style.override_colors,
        "override_radii": cluster_style.override_radii,
        "uniform_color": cluster_style.uniform_color,
        "uniform_radius": cluster_style.uniform_radius,
    }
    # Remove None values to avoid overriding function defaults
    cluster_kwargs = {k: v for k, v in cluster_kwargs.items() if v is not None}

    # Step 5: Draw the cluster using the projected coordinates
    draw_cluster(projected_df, ax, **cluster_kwargs)

    # Step 6: Optionally draw bonds
    if draw_bonds_flag:
        # Get bonds based on bond style
        bonds = get_bond_list(
            projected_df,
            bond_type=bond_style.bond_type,
            distance_cutoff=bond_style.distance_cutoff,
        )

        bond_kwargs = {
            "bonds": bonds,
            "x_column": bond_style.x_column,
            "y_column": bond_style.y_column,
            "color": bond_style.color,
            "width": bond_style.width,
            "alpha": bond_style.alpha,
            "style": bond_style.style,
        }

        draw_bonds(projected_df, ax, **bond_kwargs)

    # set axes limits with a margin to prevent clipping
    xlims, ylims = _get_axes_limits(ax, multiplier=0.1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def _get_axes_limits(ax, multiplier=0.1):
    # set the axes limits so things don't get cut off
    xlim = ax.get_xlim()
    xrange = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    # Add a margin around the limits based on the range and multiplier
    xlims = (xlim[0] - multiplier * xrange, xlim[1] + multiplier * xrange)
    ylims = (ylim[0] - multiplier * yrange, ylim[1] + multiplier * yrange)
    return xlims, ylims


if __name__ == "__main__":
    # Example usage with a sample cluster DataFrame
    import pandas as pd

    # Sample cluster data
    # data = pd.read_pickle("tests/test_data/test_groundtruth_output_2.pkl")
    data = pd.read_pickle("tests/test_data/gen_2_output.pkl")
    cluster_df = pd.DataFrame(data)
    print(cluster_df)

    # Render the cluster with custom styles
    fig, ax = render_cluster(
        cluster_df,
        azimuthal_angle=45,
        tilt_angle=30,
        cluster_style=ClusterStyle(
            override_colors={"Fe": "#B44599", "O": "#2F4858"}
        ),
        bond_style=BondStyle(
            bond_type="distance_cutoff",
            distance_cutoff=2.3,
            color="gray",
            width=8,
        ),
    )
    plt.show()
