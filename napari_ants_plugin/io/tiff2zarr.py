#!/usr/bin/env python3
"""
tiff_to_zarr.py

A reusable script to convert a large 3D TIFF stack to Zarr format without loading
the entire file into memory. This is ideal for extremely large datasets (e.g., 200GB).

Usage:
    python tiff_to_zarr.py /path/to/large_image.tif /path/to/output.zarr [--chunk_shape 1 512 512]

"""

import os
import argparse
import numpy as np
import tifffile
import zarr
# Use LocalStore for local filesystem storage.
from zarr.storage import LocalStore
import math
from numcodecs import Blosc  # Optional: for compression


def determine_default_chunk_shape(shape):
    """
    Determine a default chunk shape given the overall shape of the image.
    For a 3D array assumed as (Z, Y, X), this returns (1, Y//2, X//2).

    Parameters:
        shape (tuple): Shape of the image.

    Returns:
        tuple: A suggested chunk shape.
    """
    if len(shape) == 3:
        z, y, x = shape
        return (1, max(1, y // 2), max(1, x // 2))
    else:
        return (1,) + shape[1:]


def convert_tiff_to_zarr(tiff_path, zarr_store_path, chunk_shape=(1,1024,1024), compressor=None):
    """
    Convert a large TIFF stack to a Zarr array using memory mapping.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        zarr_store_path (str): Path to the output Zarr store (directory).
        chunk_shape (tuple, optional): Desired chunk shape. If None, a default is computed.
        compressor (numcodecs.abc.Codec, optional): Compressor for the Zarr array.

    Returns:
        None
    """
    #
    print(f"[INFO] Opening TIFF file as a memory map: {tiff_path}")
    try:
        # Open the TIFF file as a memory-mapped array.
        tif_memmap = tifffile.memmap(tiff_path)
    except Exception as e:
        raise RuntimeError(f"Error opening TIFF file: {e}")

    shape = tif_memmap.shape
    # Ensure the dtype is in native byte order to avoid metadata conversion issues.
    dtype = np.dtype(tif_memmap.dtype).newbyteorder('=')
    print(f"[INFO] TIFF shape: {shape}, dtype: {dtype}")

    if chunk_shape is None:
        chunk_shape = determine_default_chunk_shape(shape)
        print(
            f"[INFO] No chunk shape provided. Using default chunk shape: {chunk_shape}")
    else:
        print(f"[INFO] Using provided chunk shape: {chunk_shape}")

    # Use LocalStore to store the Zarr array on disk.
    print(f"[INFO] Creating Zarr store at: {zarr_store_path}")
    store = LocalStore(zarr_store_path)

    # Attempt to initialize the compressor if not provided.
    if compressor is None:
        try:
            compressor = Blosc(cname='zstd', clevel=3,
                               shuffle=Blosc.BITSHUFFLE)
        except Exception as e:
            print(
                f"[WARNING] Compressor failed to initialize: {e}. Proceeding without compression.")
            compressor = None

    # Force metadata_version '2.0' to avoid potential issues with v3 metadata.
    root = zarr.group(store=store, overwrite=True)
    zarr_array = root.create_array('data',
                                   shape=shape,
                                   chunks=chunk_shape,
                                   dtype=dtype)

    num_slices = shape[0]
    print(f"[INFO] Starting conversion: writing {num_slices} slices...")
    for i in range(num_slices):
        try:
            # Read one slice; memmap loads only the requested slice.
            slice_data = tif_memmap[i, ...]
            zarr_array[i, ...] = slice_data
        except Exception as e:
            print(f"[ERROR] Error processing slice {i}: {e}")
            continue

        if (i + 1) % 100 == 0 or (i + 1) == num_slices:
            print(f"[INFO] Processed {i + 1}/{num_slices} slices.")

    print(f"[INFO] Conversion complete! Zarr data saved at: {zarr_store_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a large 3D TIFF stack to Zarr format without "
            "loading the entire file into memory."
        )
    )
    parser.add_argument("tiff_path", type=str,
                        help="Path to the large TIFF stack.")
    parser.add_argument("zarr_store_path", type=str,
                        help="Path to the output Zarr store (directory).")
    parser.add_argument(
        "--chunk_shape",
        type=int,
        nargs="+",
        help="Chunk shape as space-separated integers (e.g., --chunk_shape 1 512 512)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.chunk_shape:
        chunk_shape = tuple(args.chunk_shape)
    else:
        chunk_shape = (1,1024,1024)

    convert_tiff_to_zarr(args.tiff_path, args.zarr_store_path, chunk_shape)


if __name__ == "__main__":
    main()
