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

    def apply_transformation(self, R, t):
        """Apply a rotation and translation to the cluster coordinates."""
        coords = self[["x", "y", "z"]].values
        transformed_coords = (R @ coords.T).T + t
        self[["x", "y", "z"]] = transformed_coords

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

    def align_with(self, reference_cdf, mask=True, allow_reflection=True):
        """Permute and align the cluster to match the order and orientation of
        a reference ClusterDataFrame."""
        from clusterrender.transform.align import align_clusters

        # Ensure both DataFrames have the same number of atoms
        if len(self) != len(reference_cdf):
            raise ValueError(
                "Both ClusterDataFrames must have the same number of atoms \
                    for alignment."
            )

        # make sure both clusters are centered
        # and sorted by distance from origin
        self.center_cluster(center_index=0)
        reference_cdf.center_cluster(center_index=0)

        # if mask is True and the reference cluster has shell information,
        # only align the center and the nearest neighbors
        if mask and "shell" in reference_cdf.columns:
            # only align the center (at origin) and nearest neighbors (shell 1)
            reference_cdf.subcluster = reference_cdf[
                reference_cdf["shell"] <= 1
            ].copy()
            self.subcluster = self[reference_cdf["shell"] <= 1].copy()
            print(
                f"Aligning {len(self.subcluster)} atoms in subcluster \
                    (center + nearest neighbors) to reference cluster."
            )
        else:
            # use the full cluster for alignment
            reference_cdf.subcluster = reference_cdf.copy()
            self.subcluster = self.copy()
            print(
                f"Aligning all {len(self.subcluster)} atoms in cluster to \
                    reference cluster."
            )

        # align the subcluster to the reference subcluster
        self.subcluster, R, t = align_clusters(
            self.subcluster,
            reference_cdf.subcluster,
            need_permute=True,
            allow_reflection=allow_reflection,
        )

        # apply the same transformation to the full cluster
        self.apply_transformation(R, t)

        return R, t

    def render(
        self,
        azimuthal_angle=45,
        tilt_angle=30,
        draw_bonds_flag=True,
        cluster_style=None,
        bond_style=None,
    ):
        """Render this cluster using the render_cluster function.

        Parameters
        ----------
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
            The rendered figure and axes with the projected visualization.

        Examples
        --------
        >>> # Basic usage with defaults
        >>> fig, ax = cluster_df.render(45, 30)

        >>> # Custom atom styling with uniform properties
        >>> from clusterrender.visualize.render import ClusterStyle, BondStyle
        >>> custom_atoms = ClusterStyle(scale=300, alpha=0.7,
        ...     uniform_color='red')
        >>> fig, ax = cluster_df.render(45, 30, cluster_style=custom_atoms)

        >>> # Custom colors by species using dictionary
        >>> atom_colors = ClusterStyle(override_colors={'Fe': 'red',
        ...     'O': 'blue'})
        >>> fig, ax = cluster_df.render(45, 30, cluster_style=atom_colors)
        """
        from clusterrender.visualize.render import render_cluster

        return render_cluster(
            cluster=self,
            azimuthal_angle=azimuthal_angle,
            tilt_angle=tilt_angle,
            draw_bonds_flag=draw_bonds_flag,
            cluster_style=cluster_style,
            bond_style=bond_style,
        )

    def render_with(
        self,
        ref_cluster,
        azimuthal_angle=45,
        tilt_angle=30,
        draw_bonds_flag=True,
        cluster_style=None,
        bond_style=None,
        ref_cluster_style=None,
    ):
        """Render this cluster with a reference cluster for comparison.

        This method renders two clusters for comparison.
        The reference cluster is rendered as outlines only while this cluster
        is rendered with full styling on top.

        Parameters
        ----------
        ref_cluster : pandas.DataFrame or ClusterDataFrame
            The reference cluster to be rendered as outlines for comparison.
        azimuthal_angle : float
            The angle in degrees defining the normal vector in the XY plane
            for projection.
        tilt_angle : float
            The angle in degrees defining the tilt of the normal vector
            from the Z axis for projection.
        draw_bonds_flag : bool, default True
            Whether to draw bonds between atoms in the main cluster.
        cluster_style : ClusterStyle, optional
            Styling parameters for this main cluster.
            If None, uses default ClusterStyle().
        bond_style : BondStyle, optional
            Styling parameters for bonds in this main cluster.
            If None, uses default BondStyle().
        ref_cluster_style : ClusterStyle, optional
            Styling parameters for the reference cluster.
            If None, uses black outlines by default.

        Returns
        -------
        fig, ax : matplotlib figure and axes
            The rendered figure and axes with both clusters.

        Examples
        --------
        >>> # Basic usage with defaults
        >>> fig, ax = cluster_df.render_with(ref_cluster, 45, 30)

        >>> # Custom styling for both clusters
        >>> from clusterrender.visualize.render import ClusterStyle, BondStyle
        >>> main_style = ClusterStyle(override_colors={'Fe': 'red',
        ...     'O': 'blue'})
        >>> ref_style = ClusterStyle(uniform_color='gray', alpha=0.5)
        >>> fig, ax = cluster_df.render_with(
        ...     ref_cluster, 45, 30,
        ...     cluster_style=main_style,
        ...     ref_cluster_style=ref_style
        ... )
        """
        from clusterrender.visualize.render import render_cluster_overlap

        return render_cluster_overlap(
            cluster=self,
            ref_cluster=ref_cluster,
            azimuthal_angle=azimuthal_angle,
            tilt_angle=tilt_angle,
            draw_bonds_flag=draw_bonds_flag,
            cluster_style=cluster_style,
            bond_style=bond_style,
            ref_cluster_style=ref_cluster_style,
        )

    def to_xyz(self, file_path, comment=""):
        """Export the cluster to an XYZ file.

        Parameters
        ----------
        file_path: str
            Path to the output XYZ file.
        """
        with open(file_path, "w") as f:
            f.write(f"{len(self)}\n")
            f.write(f"XYZ coordinates {comment}\n")
            for _, row in self.iterrows():
                f.write(f"{row['species']} {row['x']} {row['y']} {row['z']}\n")

    def assign_shells(self, cutoff_distance=2.2):
        """Assign shells to atoms based on a cutoff distance. The first row is
        the central atom (shell = 0). The following rows with distance from
        origin <= cutoff_distance has shell = 1. Every other row has shell = 2.

        Parameters
        ----------
        cutoff_distance : float
            The distance cutoff to determine which atoms belong to
            the same shell.

        Returns
        -------
        None
            The method updates the cluster dataframe in place, assigning a
            'shell' column to each atom indicating its shell number.
        """
        if "distance" not in self.columns:
            self["distance"] = (
                self["x"] ** 2 + self["y"] ** 2 + self["z"] ** 2
            ) ** 0.5

        if "shell" in self.columns:
            print("Shell column already exists.")
            return

        # sort by distance first
        self.sort_values(by="distance", inplace=True)
        # initialize a new shell column
        self["shell"] = 2
        self.loc[self["distance"] <= cutoff_distance, "shell"] = 1
        # set shell of center atom = 0
        if len(self) > 0:
            self.at[0, "shell"] = 0
        return

    def calc_difference(self, ref_cluster, metric="rmsd"):
        """Calculate the difference between this cluster and a reference
        cluster using a specified metric.

        Parameters
        ----------
        ref_cluster : ClusterDataFrame
            The reference cluster to compare against.
        metric : str, default "rmsd"
            The metric to use for calculating the difference. Options are:
            - "rmsd": Root Mean Square Distance
            - "mde": Mean Distance Error

        Returns
        -------
        float
            The calculated difference between the two clusters based on the
            specified metric.

        Raises
        ------
        ValueError
            If an unsupported metric is specified.
        """
        if metric == "rmsd":
            from clusterrender.compute.rmsd import root_mean_square_distance

            return root_mean_square_distance(self, ref_cluster)

        elif metric == "mde":
            from clusterrender.compute.mde import mean_distance_error

            return mean_distance_error(self, ref_cluster)

        else:
            raise ValueError(
                f"Unsupported metric '{metric}'. Use 'rmsd' or 'mde'."
            )

    def permute_to_match(self, ref_cluster):
        """Permute the order of atoms in this cluster to match the order of a
        reference cluster using the Hungarian algorithm.

        Parameters
        ----------
        ref_cluster : ClusterDataFrame
            The reference cluster to match.

        Returns
        -------
        None
            The method updates the cluster dataframe in place, permuting the
            rows to best match the reference cluster.
        """
        from clusterrender.transform.permute_hungarian import permute_hungarian

        permuted_cluster = permute_hungarian(self, ref_cluster)
        self.update(permuted_cluster)


if __name__ == "__main__":

    # try this first
    ref_cdf = ClusterDataFrame(
        pd.read_pickle("tests/test_data/test_groundtruth_output_2.pkl")
    )
    cdf = ClusterDataFrame(pd.read_pickle("tests/test_data/gen_2_output.pkl"))
    cdf.center_cluster(center_species="Fe")
    cdf.assign_shells(cutoff_distance=2.1)
    print(ref_cdf)
    print(cdf)
    """# rendering style import matplotlib.pyplot as plt from
    clusterrender.visualize.render import BondStyle

    bond_style = BondStyle(     bond_type="distance_cutoff",
    distance_cutoff=2.1,     color="gray",     width=8,     alpha=0.8, )

    cdf.render_with(ref_cdf, 45, 0, bond_style=bond_style) R, t =
    cdf.align_with(ref_cdf, mask=True) print(cdf) cdf.render_with(
    ref_cdf, azimuthal_angle=15, tilt_angle=15, bond_style=bond_style )
    cdf.to_xyz("output.xyz", comment="Aligned cluster test")

    plt.show()
    """
