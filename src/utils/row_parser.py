import pandas as pd
import numpy as np


def parse_groundtruth_row(row, central_atom):
    # initialize a DataFrame with the central atom
    df_cluster = _add_origin_atom(central_atom)
    # extract columns in row that contains "nn_xyz" and "nn_species"
    nn_xyz_cols = sorted([col for col in row.index if "nn_xyz" in col])
    nn_species_cols = sorted([col for col in row.index if "nn_species" in col])
    # check if the number of nn_xyz and nn_species columns match
    print(f"Found nn_xyz columns: {nn_xyz_cols}")
    print(f"Found nn_species columns: {nn_species_cols}")

    if len(nn_xyz_cols) != len(nn_species_cols):
        raise ValueError("Mismatched number of nn_xyz and nn_species columns")

    # extract columns in row that contains "nn_xyz" and "nn_species"
    nn_xyz_cols = sorted([col for col in row.index if "nn_xyz" in col])
    nn_species_cols = sorted([col for col in row.index if "nn_species" in col])
    # check if the number of nn_xyz and nn_species columns match
    print(f"Found nn_xyz columns: {nn_xyz_cols}")
    print(f"Found nn_species columns: {nn_species_cols}")

    if len(nn_xyz_cols) != len(nn_species_cols):
        raise ValueError("Mismatched number of nn_xyz and nn_species columns")

    # loop through each neighbor column and add to the DataFrame
    # shell number is determined by the order of the columns
    # (nn_xyz_1, nn_xyz_2, ...)
    for i, (xyz_col, species_col) in enumerate(
        zip(nn_xyz_cols, nn_species_cols)
    ):
        shell_number = i + 1  # shell number starts from 1
        # extract the neighbor's species and coordinates
        species = row[species_col]
        xyz_array = np.array(row[xyz_col])
        if species is None or xyz_array is None:
            print(
                f"Skipping columns {xyz_col} and {species_col} \
                    due to missing data."
            )
            continue
        try:
            df_ = pd.DataFrame(
                {
                    "species": species,
                    "x": xyz_array[:, 0],
                    "y": xyz_array[:, 1],
                    "z": xyz_array[:, 2],
                    "shell": shell_number,
                }
            )
            # append df_ to df_cluster
            df_cluster = pd.concat([df_cluster, df_], ignore_index=True)
            continue
        except Exception as e:
            print(f"Error processing columns {xyz_col} and {species_col}: {e}")
            continue

    # add distance column and sort by distance (ascending)
    df_cluster["distance"] = np.sqrt(
        df_cluster["x"] ** 2 + df_cluster["y"] ** 2 + df_cluster["z"] ** 2
    )
    df_cluster = df_cluster.sort_values(by="distance").reset_index(drop=True)
    # add site_name column (species_shell_index)
    df_cluster["site_name"] = df_cluster.apply(
        lambda row: f"{row['species']}_{row['shell']}_{row.name}", axis=1
    )

    return df_cluster


def _add_origin_atom(central_atom):
    """Add the origin atom (central atom) to the cluster DataFrame."""
    df_origin = pd.DataFrame(
        {"species": [central_atom], "x": [0], "y": [0], "z": [0], "shell": [0]}
    )
    return df_origin
