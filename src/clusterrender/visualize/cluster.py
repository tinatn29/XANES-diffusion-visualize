import numpy as np
import matplotlib.colors
from clusterrender.styles.style import get_clusterdataframe_styles
from clusterrender.visualize.atom import draw_atom

"""
Render a cluster of atoms as spheres
with colors and radii based on their element types.
"""


def draw_cluster(
    df,
    ax,
    x_column="e1",
    y_column="e2",
    species_column="species",
    scale=200,
    alpha=1.0,
    override_colors=None,
    override_radii=None,
    uniform_color=None,
    uniform_radius=None,
):
    """Draw multiple atoms from a clusterdataframe efficiently.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing atom information.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    x_column : str, default 'e1'
        Name of column containing x coordinates.
    y_column : str, default 'e2'
        Name of column containing y coordinates.
    species_column : str, default 'species'
        Name of column containing element symbols.
    scale : float, default 200
        Scaling factor for atom sizes.
    alpha : float, default 1.0
        Transparency level for all atoms.
    override_colors : dict or array-like, optional
        Custom colors for atoms. Can be:
        - dict: {species: color} mapping (e.g., {'Fe': 'red', 'O': 'blue'})
        - array-like: Custom colors for each atom.
        Shape should be (n_atoms, 3) or (n_atoms,)
    override_radii : dict or array-like, optional
        Custom radii for atoms. Can be:
        - dict: {species: radius} mapping (e.g., {'Fe': 1.2, 'O': 0.8})
        - array-like: Custom radii for each atom. Shape should be (n_atoms,)
    uniform_color : array-like, optional
        Single color to use for all atoms (overrides per-element colors).
    uniform_radius : float, optional
        Single radius to use for all atoms (overrides per-element radii).

    Returns
    -------
    None

    Notes
    -----
    This function is more efficient than calling draw_atom repeatedly
    as it fetches element styles for unique species only once.
    """
    if len(df) == 0:
        return

    # Get coordinates
    x_coords = df[x_column].values
    y_coords = df[y_column].values

    # Handle uniform overrides
    if uniform_color is not None and uniform_radius is not None:
        # Use uniform values for all atoms
        colors = (
            np.tile(uniform_color, (len(df), 1))
            if len(np.array(uniform_color).shape) == 1
            else np.array([uniform_color] * len(df))
        )
        radii = np.full(len(df), uniform_radius)
    else:
        # Get element-specific styles as base
        style_colors, style_radii = get_clusterdataframe_styles(
            df, species_column
        )

        # Start with default colors and radii
        colors = style_colors.copy()
        radii = style_radii.copy()

        # Apply overrides based on type (dict or array)
        if override_colors is not None:
            if isinstance(override_colors, dict):
                # Dict mapping: {species: color}
                species_list = df[species_column].values
                for i, species in enumerate(species_list):
                    if species in override_colors:
                        new_color = override_colors[species]
                        if isinstance(new_color, str):
                            colors[i] = matplotlib.colors.to_rgb(new_color)
                        else:
                            colors[i] = new_color
            else:
                # Array-like override (backward compatibility)
                colors = np.array(override_colors)

        if override_radii is not None:
            if isinstance(override_radii, dict):
                # Dict mapping: {species: radius}
                species_list = df[species_column].values
                for i, species in enumerate(species_list):
                    if species in override_radii:
                        radii[i] = override_radii[species]
            else:
                # Array-like override (backward compatibility)
                radii = np.array(override_radii)

        # Apply uniform overrides (highest priority)
        if uniform_color is not None:
            colors = (
                np.tile(uniform_color, (len(df), 1))
                if len(np.array(uniform_color).shape) == 1
                else np.array([uniform_color] * len(df))
            )
        if uniform_radius is not None:
            radii = np.full(len(df), uniform_radius)

    # Draw each atom
    for i, (x, y, color, radius) in enumerate(
        zip(x_coords, y_coords, colors, radii)
    ):
        # Use the single atom function for consistent rendering
        species = df.iloc[i][
            species_column
        ]  # Get species for potential debugging
        draw_atom(
            x,
            y,
            species,
            ax,
            scale=scale,
            alpha=alpha,
            override_color=color,
            override_radius=radius,
        )
    return


def draw_cluster_vectorized(
    df,
    ax,
    x_column="e1",
    y_column="e2",
    species_column="species",
    alpha=1.0,
    scale=200,
    n_layers=10,
    override_colors=None,
    override_radii=None,
    uniform_color=None,
    uniform_radius=None,
):
    """Draw multiple atoms using vectorized operations for better performance.

    This is a faster alternative to draw_cluster that sacrifices some rendering
    quality (fewer layers) for speed when drawing many atoms.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing atom information.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    x_column : str, default 'e1'
        Name of column containing x coordinates.
    y_column : str, default 'e2'
        Name of column containing y coordinates.
    species_column : str, default 'species'
        Name of column containing element symbols.
    alpha : float, default 1.0
        Transparency level for all atoms.
    scale : float, default 200
        Scaling factor for atom sizes.
    n_layers : int, default 10
        Number of layers for 3D effect (fewer = faster).
    override_colors : dict or array-like, optional
        Custom colors for atoms. Can be:
        - dict: {species: color} mapping (e.g., {'Fe': 'red', 'O': 'blue'})
        - array-like: Custom colors for each atom.
        Shape should be (n_atoms, 3) or (n_atoms,)
    override_radii : dict or array-like, optional
        Custom radii for atoms. Can be:
        - dict: {species: radius} mapping (e.g., {'Fe': 1.2, 'O': 0.8})
        - array-like: Custom radii for each atom. Shape should be (n_atoms,)
    uniform_color : array-like, optional
        Single color to use for all atoms (overrides per-element colors).
    uniform_radius : float, optional
        Single radius to use for all atoms (overrides per-element radii).

    Returns
    -------
    None

    Notes
    -----
    This function is optimized for speed when rendering many atoms.
    For highest quality rendering, use draw_atoms instead.
    """
    if len(df) == 0:
        return

    # Get coordinates and base styles
    x_coords = df[x_column].values
    y_coords = df[y_column].values
    colors, radii = get_clusterdataframe_styles(df, species_column)

    # Apply overrides based on type (dict or array)
    if override_colors is not None:
        if isinstance(override_colors, dict):
            # Dict mapping: {species: color}
            species_list = df[species_column].values
            for i, species in enumerate(species_list):
                if species in override_colors:
                    new_color = override_colors[species]
                    if isinstance(new_color, str):
                        colors[i] = matplotlib.colors.to_rgb(new_color)
                    else:
                        colors[i] = new_color
        else:
            # Array-like override
            colors = np.array(override_colors)

    if override_radii is not None:
        if isinstance(override_radii, dict):
            # Dict mapping: {species: radius}
            species_list = df[species_column].values
            for i, species in enumerate(species_list):
                if species in override_radii:
                    radii[i] = override_radii[species]
        else:
            # Array-like override
            radii = np.array(override_radii)

    # Apply uniform overrides (highest priority)
    if uniform_color is not None:
        colors = (
            np.tile(uniform_color, (len(df), 1))
            if len(np.array(uniform_color).shape) == 1
            else np.array([uniform_color] * len(df))
        )
    if uniform_radius is not None:
        radii = np.full(len(df), uniform_radius)

    # Calculate base sizes
    base_sizes = scale * np.sqrt(radii)

    # Draw layers vectorized
    for layer in range(n_layers):
        # Calculate layer properties
        size_factor = 1 - layer / n_layers
        lighten_factor = (layer / n_layers) ** 6
        layer_sizes = base_sizes * size_factor

        # Lighten colors for this layer
        layer_colors = colors * (1 - lighten_factor) + lighten_factor

        # Draw all atoms for this layer at once
        ax.scatter(
            x_coords,
            y_coords,
            s=layer_sizes,
            c=layer_colors,
            edgecolor="none",
            zorder=10 + layer,
            alpha=alpha,
        )

    # Add outlines if alpha < 1
    if alpha < 1:
        ax.scatter(
            x_coords,
            y_coords,
            s=base_sizes,
            c="none",
            edgecolor=colors,
            linewidths=0.2,
            zorder=10 + n_layers,
            alpha=min(5 * alpha, 1.0),
        )
    return
