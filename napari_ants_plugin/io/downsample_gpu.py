#!/usr/bin/env python3
"""
downsample_gpu.py

This script uses Dask and CuPy to downsample a large 3D Zarr image so that its resolution
matches a target shape (e.g., the Allen Atlas resolution). It performs out-of-core processing,
converting each block to a GPU array via CuPy and using Dask's coarsen to perform block reduction.

Usage:
    python downsample_gpu.py --input_zarr input.zarr --output_tif downsampled.tif --target_shape 50 256 256

Arguments:
    --input_zarr     Path to the input Zarr directory.
    --output_tif    Path to the output (downsampled) tif file.
    --target_shape   Three integers (Z, Y, X) representing the desired output shape.
"""

import argparse
import numpy as np
import dask.array as da
import cupy as cp
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas, show_atlases
import brainglobe_space as bg
import tiffile as tiff
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cupyx.scipy.ndimage import zoom
from tqdm import tqdm
from dask import delayed


show_atlases()


def gpu_resize(image, target_shape, order=3):
    """Resize a 2D slice using GPU-accelerated CuPy."""
    image_gpu = cp.asarray(image)  # Move to GPU
    input_shape = image_gpu.shape
    # calculate zoom factors
    zoom_factors = (target_shape[0] / input_shape[0],
                    target_shape[1] / input_shape[1])
    # use cupyx.scipy.ndimage.zoom
    resized_gpu = zoom(image_gpu, zoom_factors, order=order)
    return cp.asnumpy(resized_gpu)  # Move back to CPU


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Downsample a large 3D Zarr image using Dask + CuPy to a target shape."
    )
    parser.add_argument("--input_zarr", type=str, required=True,
                        help="Path to the input Zarr store.")
    parser.add_argument("--output_tif", type=str,
                        required=True, help="Path to the output Zarr store.")
    parser.add_argument("--atlas_name", type=str, required=True,
                        help="Atlas name (e.g., allen_mouse_25um).")
    return parser.parse_args()


def downsample_zarr(input_zarr_path,
                    output_tiff_path,
                    input_voxel_size,
                    bg_atlas,
                    zarr_origin='ial',
                    interpolation_order=1,
                    ):

    # Start Dask GPU cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Dask GPU cluster started.")

    reference_image = bg_atlas.reference
    atlas_origin = bg_atlas.orientation
    mapped_atlas = bg.map_stack_to(atlas_origin, zarr_origin, reference_image)
    target_voxel_size = bg_atlas.resolution
    print("zarr origin ", zarr_origin)

    target_shape = mapped_atlas.shape
    print("Refrence shape: ", target_shape)

    # Load the input Zarr array as a Dask array.
    print("[INFO] Loading input Zarr array...")
    group = zarr.open(input_zarr_path, mode="r")

    # Access the dataset named 'data' within the group
    zarr_array = group["data"]

    # Extract original dimensions
    large_d, large_h, large_w = zarr_array.shape
    target_d, target_h, target_w = target_shape

    # Compute XY scaling factors
    scale_h, scale_w = target_h / large_h, target_w / large_w
    print(f"XY Scaling factors: (Height={scale_h:.2f}, Width={scale_w:.2f})")

    # Convert the Zarr array to a Dask array
    dask_image = da.from_zarr(zarr_array)
    print(f"[INFO] Input shape: {dask_image.shape}")

    # --- XY Downsampling ---
    # Use NumPy array for intermediate, adjust dtype if needed
    xy_downsampled_image = np.zeros(
        (large_d, target_h, target_w), dtype=zarr_array.dtype)
    print(f"{xy_downsampled_image.shape=}")

    print("Downsampling in XY...")
    for i in tqdm(range(large_d), desc="XY Downsampling Depth Slices"):
        slice_data = dask_image[i].compute()
        xy_downsampled_slice = gpu_resize(
            slice_data, (target_h, target_w), order=interpolation_order)
        xy_downsampled_image[i, :, :] = xy_downsampled_slice

    print(
        f"XY Downsampled intermediate image shape: {xy_downsampled_image.shape}")

    # --- Z Downsampling ---
    downsampled_image = np.zeros(target_shape, dtype=zarr_array.dtype)

    # Compute Z scaling factor
    scale_d = target_d / large_d
    print(f"Z Scaling factor: (Depth={scale_d:.2f})")

    print("Downsampling in Z...")
    for j in tqdm(range(target_h), desc="Z Downsampling Height Pixels"):
        for k in range(target_w):
            # Extract depth profile for each (j, k) position
            depth_profile = xy_downsampled_image[:, j, k]
            depth_profile_resized = gpu_resize(depth_profile.reshape(
                # Resize depth profile
                -1, 1), (target_d, 1), order=interpolation_order).flatten()
            downsampled_image[:, j, k] = depth_profile_resized

    downsampled_image = bg.map_stack_to(
        zarr_origin, atlas_origin, downsampled_image)

    print(f"{downsampled_image.shape}")
    # Save the downsampled image as a TIFF file
    tiff.imwrite(output_tiff_path, downsampled_image)
    print(
        f"✅ GPU-Accelerated Two-Stage Downsampling complete! Saved at: {output_tiff_path}")

    # Shutdown Dask GPU cluster
    client.close()
    cluster.close()
    print("Dask GPU cluster closed.")


def main():
    args = parse_arguments()
    bg_atlas = BrainGlobeAtlas(args.atlas_name)  # "kim_mouse_10um")
    # Define voxel sizes in micrometers (Z, Y, X)
    input_voxel_size = (4.0, 2.0, 2.0)  # For example: Z: 4um, Y: 2um, X: 2um
    downsample_zarr(args.input_zarr,
                    args.output_tif,
                    input_voxel_size,
                    bg_atlas=bg_atlas)


if __name__ == "__main__":
    main()
