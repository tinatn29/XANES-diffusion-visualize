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

        # we might use this to only transform nearest neighbors
        self.subcluster = None

    @classmethod
    def from_groundtruth_row(cls, row, central_atom):
        """Create a ClusterDataFrame from a row of a ground truth DataFrame."""
        from utils.row_parser import parse_groundtruth_row

        df = parse_groundtruth_row(row, central_atom)
        return cls(df, central_atom=central_atom)

    @classmethod
    def from_xyz(cls, file_path, center_species=None, center_index=None):
        """Create a ClusterDataFrame from an XYZ file."""
        from utils.xyz_parser import parse_xyz_file

        df = parse_xyz_file(file_path)
        cdf = cls(df)
        # center the cluster if center_species or center_index is specified
        if center_species is not None or center_index is not None:
            cdf.center_cluster(
                center_species=center_species, center_index=center_index
            )
        return cdf

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

    def center_cluster(self, center_species=None, center_index=None):
        """Center the cluster on a specified atom by translating all
        coordinates.

        Output is sorted by distance from the new origin, with the new
        center atom as the first row.
        """
        if center_index is not None:
            if center_index < 0 or center_index >= len(self):
                raise IndexError("Index out of bounds for cluster.")
            center_atom = self.iloc[center_index]
        elif center_species is not None:
            # sort everything by distance first
            if "distance" not in self.columns:
                self["distance"] = np.sqrt(
                    self["x"] ** 2 + self["y"] ** 2 + self["z"] ** 2
                )
            self.sort_values(by="distance").reset_index(
                drop=True, inplace=True
            )
            # center_atom = atom of the specified species closest to origin
            try:
                center_atom = self[self["species"] == center_species].loc[0]
            except IndexError:
                raise ValueError(
                    f"No atom with species '{center_species}' found."
                )
        else:
            raise ValueError(
                "Must specify either species or index to center on."
            )

        # translate the coordinates
        self["x"] -= center_atom["x"]
        self["y"] -= center_atom["y"]
        self["z"] -= center_atom["z"]
        # recompute distance from new origin and sort
        self["distance"] = np.sqrt(
            self["x"] ** 2 + self["y"] ** 2 + self["z"] ** 2
        )
        self.sort_values(by="distance", inplace=True)
        self.reset_index(drop=True, inplace=True)
