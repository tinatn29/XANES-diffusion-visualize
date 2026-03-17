import pandas as pd
import numpy as np


def parse_xyz_file(file_path, lines_to_skip=2):
    """Parse an XYZ file and return a ClusterDataFrame.

    Parameters
    ----------
    file_path : str
        Path to the XYZ file to be parsed.
        lines_to_skip : int, optional
        Number of lines to skip at the beginning of the file (default is 2).

    Returns
    -------
    df_cluster: pandas.DataFrame
        A DataFrame containing the species and coordinates of atoms in the XYZ
        file, sorted by distance from the origin.
        columns: species, x, y, z, distance.
    """
    cluster = {
        "species": [],
        "x": [],
        "y": [],
        "z": [],
    }  # dictionary to hold points
    with open(file_path, "r") as file:
        # Skip the first 'lines_to_skip' lines
        for _ in range(lines_to_skip):
            next(file)

        # Now read the rest of the file
        for line in file:
            parts = line.split()
            # Store species and coordinates
            element = parts[0]
            cluster["species"].append(element)
            x, y, z = map(float, parts[1:4])  # Convert coordinates to float
            cluster["x"].append(x)
            cluster["y"].append(y)
            cluster["z"].append(z)

    # convert points into a DataFrame
    df_cluster = pd.DataFrame(cluster)
    df_cluster["distance"] = np.sqrt(
        df_cluster["x"] ** 2 + df_cluster["y"] ** 2 + df_cluster["z"] ** 2
    )
    # sort by distance
    df_cluster = df_cluster.sort_values(by="distance").reset_index(drop=True)
    return df_cluster
