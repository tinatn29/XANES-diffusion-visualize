import json
import os
import numpy as np
from typing import List, Tuple, Dict, Any

"""Helper functions to retrieve element styles for rendering."""

# Cache for element styles data
_element_styles = None


def _load_element_styles() -> Dict[str, Dict[str, Any]]:
    """Load element styles from JSON file. Cached after first call.

    Returns
    -------
    dict of str : dict
        Dictionary mapping element symbols to style dictionaries.
    """
    global _element_styles
    if _element_styles is None:
        # Get path to element-style.json relative to this file
        current_dir = os.path.dirname(__file__)
        json_path = os.path.join(current_dir, "lib", "element-styles.json")

        with open(json_path, "r") as f:
            _element_styles = json.load(f)

    return _element_styles


def get_element_style(element: str) -> Dict[str, Any]:
    """Get complete style information (color and radius) for a single element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H', 'C', 'O').

    Returns
    -------
    dict
        Dictionary with 'color' (RGB list) and 'radius' (float).

    Raises
    ------
    KeyError
        If element is not found in style data.
    """
    styles = _load_element_styles()
    if element not in styles:
        raise KeyError(f"Element '{element}' not found in style data")
    return styles[element]


def get_element_styles(elements: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get complete style information for multiple elements efficiently.

    Parameters
    ----------
    elements : list of str
        List of element symbols.

    Returns
    -------
    dict of str : dict
        Dictionary mapping element symbols to their style data.

    Raises
    ------
    KeyError
        If any element is not found in style data.
    """
    styles = _load_element_styles()
    result = {}
    missing = []

    for element in elements:
        if element in styles:
            result[element] = styles[element]
        else:
            missing.append(element)

    if missing:
        raise KeyError(f"Elements not found in style data: {missing}")

    return result


def get_element_color(element: str) -> List[float]:
    """Get RGB color for a single element.

    Parameters
    ----------
    element : str
        Element symbol.

    Returns
    -------
    list of float
        RGB color as list [R, G, B] with values in [0, 1].
    """
    return get_element_style(element)["color"]


def get_element_colors(elements: List[str]) -> np.ndarray:
    """Get RGB colors for multiple elements efficiently.

    Parameters
    ----------
    elements : list of str
        List of element symbols.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_elements, 3) with RGB values.
    """
    styles = get_element_styles(elements)
    colors = np.array([styles[element]["color"] for element in elements])
    return colors


def get_element_radius(element: str) -> float:
    """Get covalent radius for a single element.

    Parameters
    ----------
    element : str
        Element symbol.

    Returns
    -------
    float
        Covalent radius in Angstroms.
    """
    return get_element_style(element)["radius"]


def get_element_radii(elements: List[str]) -> np.ndarray:
    """Get covalent radii for multiple elements efficiently.

    Parameters
    ----------
    elements : list of str
        List of element symbols.

    Returns
    -------
    numpy.ndarray
        Array of radii in Angstroms.
    """
    styles = get_element_styles(elements)
    radii = np.array([styles[element]["radius"] for element in elements])
    return radii


def get_clusterdataframe_styles(
    df, element_column: str = "species"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get colors and radii for all atoms in a clusterdataframe efficiently.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with species information.
    element_column : str, default 'species'
        Name of column containing element symbols.

    Returns
    -------
    colors : numpy.ndarray
        Array of shape (n_atoms, 3) with RGB values.
    radii : numpy.ndarray
        Array of shape (n_atoms,) with radius values.
    """
    elements = df[element_column].tolist()
    unique_elements = list(set(elements))

    # Get styles for unique elements only
    unique_styles = get_element_styles(unique_elements)

    # Create lookup arrays
    element_to_color = {
        elem: unique_styles[elem]["color"] for elem in unique_elements
    }
    element_to_radius = {
        elem: unique_styles[elem]["radius"] for elem in unique_elements
    }

    # Map to full arrays
    colors = np.array([element_to_color[elem] for elem in elements])
    radii = np.array([element_to_radius[elem] for elem in elements])

    return colors, radii
