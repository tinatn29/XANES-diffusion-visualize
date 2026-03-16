"""This module defines class ClusterDataFrame.

A local structure or cluster of atoms is represented in a DataFrame
format.
"""

import pandas as pd

# -------------------------


class ClusterDataFrame(pd.DataFrame):
    """Define a cluster of atoms in a pandas DataFrame format.

    Each row corresponds to an atom, and columns represent
    atomic properties: species, xyz coordinates, and (optionally)
    coordination shells, specifying the central atom (0) and its
    neighboring atoms (1, 2, ...).

    Methods
    -------
    parse_structure(structure_input, site_index=0)
        Parse structure data from a structure, a file, a dictionary,
        or a DataFrame into ClusterDataFrame.

    Attributes
    ----------
    _constructor : property
        Ensures that DataFrame operations return ClusterDataFrame objects.
    """

    @property
    def _constructor(self):
        return ClusterDataFrame

    def __init__(self, data=None, *args, **kwargs):
        """Initialize ClusterDataFrame with DataFrame-compatible arguments."""
        if kwargs.get("copy") is None and isinstance(data, pd.DataFrame):
            kwargs.update(copy=True)

        super().__init__(data, *args, **kwargs)

        if isinstance(data, pd.DataFrame) and data.attrs:
            self.attrs = data.attrs
