#!/usr/bin/env python3
"""
downsample_gpu.py

This script uses Dask and CuPy to downsample a large 3D Zarr image so that its resolution
matches a target shape (e.g., the Allen Atlas resolution). It performs out-of-core processing,
converting each block to a GPU array via CuPy and using Dask's distributed processing.

Usage:
    python downsample_gpu.py --input_zarr input.zarr --output_tif downsampled.tif --atlas_name allen_mouse_25um

Arguments:
    --input_zarr     Path to the input Zarr directory.
    --output_tif     Path to the output (downsampled) TIFF file.
    --atlas_name     Atlas name (e.g., allen_mouse_25um).
    --log_file       Path to the log file (default: downsample_gpu.log).
"""

import argparse
import logging
import sys
import os

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


def setup_logging(log_file="downsample_gpu.log"):
    """
    Set up logging to both console and file.

    Args:
        log_file (str): Path to the log file.
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers for console and file logging.
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Define formatters.
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Add handlers.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def gpu_resize(image, target_shape, order=3):
    """
    Resize an image (2D or 2D representation of 1D) using GPU-accelerated CuPy.

    Args:
        image (np.ndarray): Input image or 1D column vector.
        target_shape (tuple): Desired output shape.
        order (int): Interpolation order (default is 3).

    Returns:
        np.ndarray: Resized image (moved back to CPU).
    """
    image_gpu = cp.asarray(image)
    input_shape = image_gpu.shape
    # Calculate zoom factors for each dimension.
    zoom_factors = tuple(t / s for t, s in zip(target_shape, input_shape))
    resized_gpu = zoom(image_gpu, zoom_factors, order=order)
    return cp.asnumpy(resized_gpu)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Downsample a large 3D Zarr image using Dask and CuPy to a target shape."
    )
    parser.add_argument("--input_zarr", type=str, required=True,
                        help="Path to the input Zarr store.")
    parser.add_argument("--output_tif", type=str, required=True,
                        help="Path to the output TIFF file.")
    parser.add_argument("--atlas_name", type=str, required=True,
                        help="Atlas name (e.g., allen_mouse_25um).")
    parser.add_argument("--log_file", type=str, default="downsample_gpu.log",
                        help="Path to the log file.")
    return parser.parse_args()


def downsample_xy(dask_image, target_h, target_w, interpolation_order, logger):
    """
    Downsample each XY slice of the 3D image.

    Args:
        dask_image (dask.array): Input 3D image.
        target_h (int): Target height.
        target_w (int): Target width.
        interpolation_order (int): Interpolation order.
        logger (logging.Logger): Logger for output.

    Returns:
        np.ndarray: 3D image with XY downsampled.
    """
    num_slices = dask_image.shape[0]
    downsampled_xy = np.zeros(
        (num_slices, target_h, target_w), dtype=dask_image.dtype)

    logger.info(f"Starting XY downsampling on {num_slices} slices...")
    for i in tqdm(range(num_slices), desc="XY Downsampling Slices"):
        slice_data = dask_image[i].compute()
        resized_slice = gpu_resize(
            slice_data, (target_h, target_w), order=interpolation_order)
        downsampled_xy[i] = resized_slice
    logger.info("XY downsampling complete.")
    return downsampled_xy


def downsample_z(xy_image, target_d, interpolation_order, logger):
    """
    Downsample the Z-dimension of the image.

    Args:
        xy_image (np.ndarray): 3D image already downsampled in XY.
        target_d (int): Target depth.
        interpolation_order (int): Interpolation order.
        logger (logging.Logger): Logger for output.

    Returns:
        np.ndarray: Fully downsampled 3D image.
    """
    orig_d, height, width = xy_image.shape
    downsampled = np.zeros((target_d, height, width), dtype=xy_image.dtype)

    logger.info("Starting Z downsampling...")
    for j in tqdm(range(height), desc="Z Downsampling (Height)"):
        for k in range(width):
            depth_profile = xy_image[:, j, k]
            # Reshape depth_profile to a column vector for consistent processing.
            resized_profile = gpu_resize(
                depth_profile.reshape(-1, 1), (target_d, 1), order=interpolation_order)
            downsampled[:, j, k] = resized_profile.flatten()
    logger.info("Z downsampling complete.")
    return downsampled


def downsample_zarr(input_zarr_path, output_tiff_path, input_voxel_size, bg_atlas, logger,
                    zarr_origin='ial', interpolation_order=1):
    """
    Downsample a Zarr array to the target shape defined by the atlas.

    Args:
        input_zarr_path (str): Path to the input Zarr.
        output_tiff_path (str): Path to the output TIFF.
        input_voxel_size (tuple): Original voxel size (Z, Y, X).
        bg_atlas (BrainGlobeAtlas): Atlas object.
        logger (logging.Logger): Logger.
        zarr_origin (str): Orientation of the Zarr dataset.
        interpolation_order (int): Order of interpolation.
    """
    # Start the Dask GPU cluster.
    cluster = LocalCUDACluster()
    client = Client(cluster)
    logger.info("Dask GPU cluster started.")

    reference_image = bg_atlas.reference
    atlas_origin = bg_atlas.orientation
    mapped_atlas = bg.map_stack_to(atlas_origin, zarr_origin, reference_image)
    target_voxel_size = bg_atlas.resolution
    target_shape = mapped_atlas.shape

    logger.info(f"Zarr origin: {zarr_origin}")
    logger.info(f"Reference (target) shape: {target_shape}")

    # Load input Zarr array.
    logger.info("Loading input Zarr array...")
    group = zarr.open(input_zarr_path, mode="r")
    zarr_array = group["data"]
    logger.info(f"Zarr image shape: {zarr_array.shape}")
    large_d, large_h, large_w = zarr_array.shape
    target_d, target_h, target_w = target_shape

    # Compute scaling factors.
    scale_h = target_h / large_h
    scale_w = target_w / large_w
    scale_d = target_d / large_d

    logger.info(
        f"Scaling factors - Depth: {scale_d:.2f}, Height: {scale_h:.2f}, Width: {scale_w:.2f}")
    logger.debug(
        f"Command Parameters: input_zarr={input_zarr_path}, output_tiff={output_tiff_path}, "
        f"input_voxel_size={input_voxel_size}, atlas_resolution={target_voxel_size}, "
        f"interpolation_order={interpolation_order}"
    )

    if os.path.exists(output_tiff_path):
        logger.info(
            f"Output file {output_tiff_path} already exists. Skipping downsampling.")
        sys.exit(0)

    # Convert to a Dask array.
    dask_image = da.from_zarr(zarr_array)
    logger.info(f"Input image shape: {dask_image.shape}")

    # --- XY Downsampling ---
    xy_downsampled = downsample_xy(
        dask_image, target_h, target_w, interpolation_order, logger)

    # --- Z Downsampling ---
    downsampled_image = downsample_z(
        xy_downsampled, target_d, interpolation_order, logger)

    # Reorient the image from the Zarr to the atlas orientation.
    downsampled_image = bg.map_stack_to(
        zarr_origin, atlas_origin, downsampled_image)
    logger.info(f"Final downsampled image shape: {downsampled_image.shape}")

    # Save the output TIFF.
    tiff.imwrite(output_tiff_path, downsampled_image)
    logger.info(f"Downsampled image saved at: {output_tiff_path}")

    # Shutdown the Dask cluster.
    client.close()
    cluster.close()
    logger.info("Dask GPU cluster closed.")


def main():
    args = parse_arguments()
    logger = setup_logging(args.log_file)
    logger.info("Starting downsampling process...")
    show_atlases()

    try:
        bg_atlas = BrainGlobeAtlas(args.atlas_name)
    except Exception as e:
        logger.error(f"Error loading atlas '{args.atlas_name}': {e}")
        sys.exit(1)

    # Define input voxel sizes in micrometers (Z, Y, X). Adjust as needed.
    input_voxel_size = (4.0, 2.0, 2.0)
    downsample_zarr(args.input_zarr, args.output_tif,
                    input_voxel_size, bg_atlas, logger)
    logger.info("Downsampling process completed successfully.")


if __name__ == "__main__":
    main()
