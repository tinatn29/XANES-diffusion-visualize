import matplotlib.colors
import numpy as np
from clusterrender.styles.style import get_element_style

"""
Draw an atom as a sphere with 3D effect using overlapping circles.
The atom's color and size are determined by its element type,
with optional overrides.
When alpha < 1.0, an outline is added to maintain visual definition.
"""


def _lighten_color(color, factor):
    """Lighten a color by a given factor.

    Parameters
    ----------
    color : str or tuple
        Matplotlib color string or RGB tuple.
    factor : float
        Factor to lighten the color (0 = no change, 1 = white).

    Returns
    -------
    tuple
        Lightened RGB color tuple.
    """
    rgb = matplotlib.colors.to_rgb(color)
    lightened_rgb = tuple((1 - factor) * c + factor for c in rgb)
    return lightened_rgb


def _atom_size(radius, scale=200):
    """Convert atomic radius to a marker size for plotting.

    Parameters
    ----------
    radius : float
        Atomic radius in Angstroms.

    scale : float, optional
        Scaling factor to adjust the marker size (default is 200).

    Returns
    -------
    float
        Marker size for plotting.
    """
    return scale * np.sqrt(radius)


def draw_atom(
    x,
    y,
    species,
    ax,
    scale=200,
    alpha=1.0,
    override_color=None,
    override_radius=None,
):
    """Draw a single atom as a sphere with 3D effect using overlapping circles.

    Parameters
    ----------
    x : float
        X coordinate of the atom center.
    y : float
        Y coordinate of the atom center.
    species : str
        Element symbol (e.g., 'H', 'C', 'O') for styling lookup.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    scale : float, optional
        Scaling factor to adjust the marker size (default is 200).
    alpha : float, default 1.0
        Transparency level (0.0 = transparent, 1.0 = opaque).
    override_color : array-like, optional
        RGB color to use instead of element's default color.
        Should be in format [R, G, B] with values in [0, 1].
    override_radius : float, optional
        Radius to use instead of element's default covalent radius.

    Returns
    -------
    None

    Notes
    -----
    The atom is rendered as 30 concentric circles with progressively lighter
    colors and smaller sizes to create a 3D sphere effect. When alpha < 1.0,
    an outline is added to maintain visual definition.
    """

    # load atom's style
    style = get_element_style(species)
    base_color = (
        override_color if override_color is not None else style["color"]
    )
    radius = (
        override_radius if override_radius is not None else style["radius"]
    )

    # draw an atom as overlapping circles to create a 3D effect
    n_layers = 30
    base_size = _atom_size(
        radius, scale=scale
    )  # base marker size based on atom's radius

    # draw concentric circles with increasing size and lighter color
    for i in range(n_layers):
        size = base_size * (1 - i / n_layers)
        lighten = (
            i / n_layers
        ) ** 6  # Squared to increase lightness more gradually
        color = _lighten_color(base_color, lighten)
        ax.scatter(
            [x],
            [y],
            s=size,
            color=[color],
            edgecolor="none",
            zorder=10 + i,
            alpha=alpha,
        )

    # if alpha < 1: draw the outline back in
    if alpha < 1:
        ax.scatter(
            [x],
            [y],
            s=base_size,
            color="none",
            edgecolor=base_color,
            linewidths=0.2,
            zorder=10 + n_layers,
            alpha=min(5 * alpha, 1.0),
        )
    return


def draw_atom_outline(
    x, y, species, ax, scale=200, alpha=1.0, override_color=None
):
    """Draw an outline of an atom as a circle.

    Parameters
    ----------
    x : float
        X coordinate of the atom center.
    y : float
        Y coordinate of the atom center.
    species : str
        Element symbol (e.g., 'H', 'C', 'O') for styling lookup.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    scale : float, optional
        Scaling factor to adjust the marker size (default is 200).
    override_color : array-like, optional
        RGB color to use instead of element's default color.
        Should be in format [R, G, B] with values in [0, 1].

    Returns
    -------
    None
    """

    # load atom's style
    style = get_element_style(species)
    line_color = (
        override_color if override_color is not None else style["color"]
    )
    radius = style["radius"]

    # draw the outline as a circle with no fill and a colored edge
    base_size = _atom_size(radius, scale=scale)
    ax.scatter(
        [x],
        [y],
        s=base_size,
        color="none",
        edgecolor=line_color,
        linewidths=0.5,
        zorder=100,
        alpha=alpha,
    )

    return
