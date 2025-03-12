#!/usr/bin/env python3
"""
Calculate brain region volumes from a registered atlas TIFF file,
including aggregated (non-leaf) volumes (e.g., for root or 'grey') using
a tree-based recursive approach and the BrainGlobe Atlas API.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import tifffile
from typing import Tuple, Dict

from brainglobe_atlasapi import BrainGlobeAtlas


def compute_recursive_voxel_count(
    region_id: int,
    tree: any,
    leaf_counts: Dict[int, int],
    memo: Dict[int, int]
) -> int:
    """
    Recursively compute the total voxel count for a region by summing the voxel count
    of the region itself (if present) and that of all its descendants.
    """
    if region_id in memo:
        return memo[region_id]
    total = leaf_counts.get(region_id, 0)
    for child in tree.children(region_id):
        total += compute_recursive_voxel_count(
            child.identifier, tree, leaf_counts, memo)
    memo[region_id] = total
    return total


def calculate_region_volumes(
    atlas_tif_path: str,
    atlas: BrainGlobeAtlas,
    voxel_size: Tuple[float, float, float]
) -> pd.DataFrame:
    """
    Calculate brain region volumes from a registered atlas stored as a TIFF file.
    This function computes the leaf voxel counts and then aggregates volumes for
    non-leaf nodes by recursively summing the counts over the atlas tree.

    Parameters
    ----------
    atlas_tif_path : str
        Path to the registered atlas TIFF file.
    atlas : BrainGlobeAtlas
        An instantiated BrainGlobeAtlas object (e.g., for 'kim_mouse_25um') containing metadata.
    voxel_size : tuple of float
        The voxel dimensions (in mm) in (z, y, x) order.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
          - region_id: integer label from the atlas.
          - region_acronym: the region acronym.
          - region_name: anatomical name of the region.
          - aggregated_voxel_count: total voxel count (region + descendants).
          - aggregated_volume_mm3: aggregated_voxel_count * voxel_volume.
    """
    logging.info(f"Loading atlas data from: {atlas_tif_path}")
    atlas_data = tifffile.imread(atlas_tif_path)
    logging.info("Atlas data loaded successfully.")

    # Count voxels for each region (excluding background labeled as 0).
    labels, counts = np.unique(atlas_data, return_counts=True)
    logging.info("Calculated unique labels and voxel counts.")

    # Build dictionary of leaf counts: region id -> voxel count.
    leaf_counts = {
        int(label): int(count)
        for label, count in zip(labels, counts)
        if label != 0
    }
    # Calculate the volume per voxel (in mm³).
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    logging.info(f"Voxel volume: {voxel_volume:.6f} mm³")

    # Create a mapping from atlas lookup: region id -> (acronym, name)
    mapping = {
        int(row["id"]): (row["acronym"], row["name"])
        for _, row in atlas.lookup_df.iterrows()
    }

    tree = atlas.structures.tree
    memo: Dict[int, int] = {}
    aggregated_data = []

    # Loop over all regions in the atlas lookup.
    for region_id, (acronym, name) in mapping.items():
        aggregated_count = compute_recursive_voxel_count(
            region_id, tree, leaf_counts, memo)
        if aggregated_count > 0:
            aggregated_data.append({
                "region_id": region_id,
                "region_acronym": acronym,
                "region_name": name,
                "aggregated_voxel_count": aggregated_count,
                "aggregated_volume_mm3": aggregated_count * voxel_volume
            })

    df = pd.DataFrame(aggregated_data)
    logging.info("Aggregated region volume calculation complete.")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate brain region volumes (including non-leaf nodes) using a recursive tree-based approach and BrainGlobe Atlas API."
    )
    parser.add_argument(
        "--atlas-tif-path",
        type=str,
        required=True,
        help="Path to the registered atlas TIFF file."
    )
    parser.add_argument(
        "--atlas-name",
        type=str,
        default="kim_mouse_25um",
        help="Name of the atlas (default: 'kim_mouse_25um')."
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        default="0.025,0.025,0.025",
        help="Voxel size in mm (comma-separated, e.g., '0.025,0.025,0.025' for 25µm voxels)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="brain_region_volumes.csv",
        help="Output CSV file for the results."
    )
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    logging.info("Starting brain region volume calculation...")

    # Parse voxel_size from string to tuple of floats.
    voxel_size = tuple(map(float, args.voxel_size.split(",")))
    logging.info(f"Voxel size set to: {voxel_size}")

    # Initialize the BrainGlobe Atlas API with check_latest set to False.
    atlas = BrainGlobeAtlas(args.atlas_name, check_latest=False)
    logging.info(f"Initialized BrainGlobe Atlas for: {args.atlas_name}")

    # Calculate aggregated region volumes.
    df_volumes = calculate_region_volumes(
        args.atlas_tif_path, atlas, voxel_size)
    logging.info("Volumes calculated successfully:")
    logging.info(df_volumes.to_string(index=False))

    # Save the results to CSV.
    df_volumes.to_csv(args.output_csv, index=False)
    logging.info(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
