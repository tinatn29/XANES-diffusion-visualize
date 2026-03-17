"""This module defines class ClusterDataFrame.

A local structure or cluster of atoms is represented in a DataFrame
format.
"""

import pandas as pd
import numpy as np

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

    def __init__(self, data=None, central_atom=None, *args, **kwargs):
        """Initialize ClusterDataFrame with DataFrame-compatible arguments."""
        if kwargs.get("copy") is None and isinstance(data, pd.DataFrame):
            kwargs.update(copy=True)

        super().__init__(data, *args, **kwargs)

        if isinstance(data, pd.DataFrame) and data.attrs:
            self.attrs = data.attrs

        if central_atom is not None:
            self.central_atom = central_atom

    @classmethod
    def from_groundtruth_row(cls, row, central_atom):
        """Create a ClusterDataFrame from a row of a ground truth DataFrame."""
        from utils.row_parser import parse_groundtruth_row

        df = parse_groundtruth_row(row, central_atom)
        return cls(df, central_atom=central_atom)

    @classmethod
    def from_xyz(cls, file_path):
        """Create a ClusterDataFrame from an XYZ file."""
        from utils.xyz_parser import parse_xyz_file

        df = parse_xyz_file(file_path)
        return cls(df)

    def add_closest_lower_shell_neighbor(self):
        """Add a column for the closest lower shell neighbor."""
        if "shell" not in self.columns:
            raise ValueError(
                "ClusterDataFrame must have 'shell' column to add \
                    closest lower shell neighbor."
            )

        if "site_name" not in self.columns:
            # add site_name column if it doesn't exist
            # sort first by distance to ensure correct site_name assignment
            self["distance"] = (
                self["x"] ** 2 + self["y"] ** 2 + self["z"] ** 2
            ) ** 0.5
            self.sort_values(by="distance").reset_index(
                drop=True, inplace=True
            )
            self["site_name"] = self.apply(
                lambda row: f"{row['species']}_{row['shell']}_{row.name}",
                axis=1,
            )

        def find_closest_lower_shell_neighbor(row):
            if row["shell"] == 0:
                return "center"
            else:
                lower_shell_neighbors = self[self["shell"] == row["shell"] - 1]

                if lower_shell_neighbors.empty:
                    return None
                elif len(lower_shell_neighbors) == 1:
                    return lower_shell_neighbors.iloc[0]["site_name"]
                else:
                    distances = np.sqrt(
                        (lower_shell_neighbors["x"] - row["x"]) ** 2
                        + (lower_shell_neighbors["y"] - row["y"]) ** 2
                        + (lower_shell_neighbors["z"] - row["z"]) ** 2
                    )
                    closest_neighbor_index = distances.idxmin()
                    return lower_shell_neighbors.loc[closest_neighbor_index][
                        "site_name"
                    ]

        self["nearest_lower_shell_neighbor"] = self.apply(
            find_closest_lower_shell_neighbor, axis=1
        )
