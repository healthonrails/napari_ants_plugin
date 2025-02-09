#!/usr/bin/env python3
"""
downsample_zarr.py

This script uses Dask and CuPy to downsample a large 3D Zarr image so that its resolution
matches a target shape (e.g., the Allen Atlas resolution). It performs out-of-core processing,
converting each block to a GPU array via CuPy and using Dask's coarsen to perform block reduction.

Usage:
    python downsample_zarr.py --input_zarr input.zarr --output_zarr downsampled.zarr --target_shape 50 256 256

Arguments:
    --input_zarr     Path to the input Zarr directory.
    --output_zarr    Path to the output (downsampled) Zarr directory.
    --target_shape   Three integers (Z, Y, X) representing the desired output shape.
"""

import argparse
import numpy as np
import dask.array as da
import cupy as cp
import zarr
from zarr.storage import LocalStore
from brainglobe_atlasapi import BrainGlobeAtlas, show_atlases
import brainglobe_space as bg

show_atlases()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Downsample a large 3D Zarr image using Dask + CuPy to a target shape."
    )
    parser.add_argument("--input_zarr", type=str, required=True,
                        help="Path to the input Zarr store.")
    parser.add_argument("--output_zarr", type=str,
                        required=True, help="Path to the output Zarr store.")
    parser.add_argument("--atlas_name", type=str, required=True,
                        help="Atlas name (e.g., allen_mouse_25um).")
    return parser.parse_args()


def compute_downsampling_factors(input_shape, target_shape, input_voxel_size, target_voxel_size):
    """
    Compute integer downsampling factors for each axis based on voxel sizes.
    """
    factors = []
    for inp_size, targ_size, inp_vox, targ_vox in zip(input_shape, target_shape, input_voxel_size, target_voxel_size):
        # factor = int(round((inp_size * inp_vox) / (targ_size * targ_vox)))
        factor = int(round(targ_vox / inp_vox))
        factor = max(factor, 1)  # Ensure factor is at least 1
        factors.append(factor)
    return tuple(factors)


def downsample_zarr(input_zarr_path,
                    output_zarr_path,
                    input_voxel_size,
                    bg_atlas,
                    zarr_origin='ial',
                    ):

    reference_image = bg_atlas.reference
    atlas_origin = bg_atlas.orientation
    target_voxel_size = bg_atlas.resolution
    print("zarr origin ", zarr_origin)

    target_shape = reference_image.shape
    print("Refrence shape: ", target_shape)

    # Load the input Zarr array as a Dask array.
    print("[INFO] Loading input Zarr array...")
    group = zarr.open(input_zarr_path, mode="r")

    # Access the dataset named 'data' within the group
    zarr_array = group["data"]

    # Convert the Zarr array to a Dask array
    darr = da.from_zarr(zarr_array)
    print(f"[INFO] Input shape: {darr.shape}")

    # Convert each block to a CuPy array so that subsequent computations run on the GPU.
    darr_gpu = darr.map_blocks(
        cp.asarray, dtype=darr.dtype, meta=cp.zeros((1, 1, 1), dtype=darr.dtype))

    # Compute downsampling factors based on the target shape and voxel sizes
    factors = compute_downsampling_factors(
        darr_gpu.shape, target_shape, input_voxel_size, target_voxel_size)
    print(f"[INFO] Computed downsampling factors: {factors}")

    # Downsample using Dask's coarsen with a CuPy reduction (here, we use cp.mean for block averaging).
    factor_dict = {axis: factor for axis, factor in enumerate(factors)}
    downsampled = da.coarsen(cp.mean, darr_gpu, factor_dict, trim_excess=True)
    print(f"[INFO] Downsampled shape (before compute): {downsampled.shape}")

    # Trigger computation. The result will be a CuPy array; convert it back to NumPy.
    downsampled_result = downsampled.compute()
    downsampled_result = cp.asnumpy(downsampled_result)
    print(f"[INFO] Final downsampled shape: {downsampled_result.shape}")
    downsampled_result = bg.map_stack_to(
        zarr_origin, atlas_origin, downsampled_result)
    

    # Save the downsampled result to a new Zarr store using LocalStore.
    print(f"[INFO] Saving downsampled image to {output_zarr_path} ...")
    store = LocalStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    _zarr_array = root.create_array(
        'data', shape=downsampled_result.shape, dtype=downsampled_result.dtype)
    _zarr_array[:] = downsampled_result

    print("[INFO] Downsampling complete and saved.")


def main():
    args = parse_arguments()
    bg_atlas = BrainGlobeAtlas(args.atlas_name)  # "kim_mouse_10um")
    # Define voxel sizes in micrometers (Z, Y, X)
    input_voxel_size = (4.0, 2.0, 2.0)  # For example: Z: 4um, Y: 2um, X: 2um
    downsample_zarr(args.input_zarr,
                    args.output_zarr,
                    input_voxel_size,
                    bg_atlas=bg_atlas)


if __name__ == "__main__":
    main()
