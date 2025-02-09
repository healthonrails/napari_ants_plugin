#!/usr/bin/env python3
"""
register_atlas_to_zarr.py

This script loads a downsampled image stored in Zarr format,
fetches atlas template and annotation via the BrainGlobe Atlas API using an atlas name (e.g., "allen_mouse_25um"),
registers the atlas (as the moving image) to the downsampled image (as the fixed image) using ANTsPy,
and applies the computed transform to the atlas annotation.
The resulting transformed annotation is saved to the specified output path.

Usage:
    python register_atlas_to_zarr.py \
      --downsampled_zarr path/to/downsampled.zarr \
      --atlas_name allen_mouse_25um \
      --output_annotation transformed_annotation.nii.gz
"""

import argparse
import dask.array as da
import zarr
import numpy as np
import ants
from brainglobe_atlasapi import BrainGlobeAtlas  # BrainGlobe Atlas API
import brainglobe_space as bg
import tifffile as tiff
import os
import shutil


def load_downsampled_image_from_zarr(zarr_path):
    """
    Load the downsampled image from a Zarr store using Dask
    and convert it to an ANTs image.
    """
    print(f"[INFO] Loading downsampled image from Zarr store: {zarr_path}")
    group = zarr.open(zarr_path, mode='r')
    zarr_arry = group['data']
    darr = da.from_zarr(zarr_arry)
    print(f"[INFO] Downsampled image shape (from Zarr): {darr.shape}")
    np_img = darr.compute()
    # Create an ANTs image from the numpy array.
    ant_img = ants.from_numpy(np_img)
    return ant_img


def load_atlas_images(atlas_name):
    """
    Use the BrainGlobe Atlas API to retrieve the atlas template and annotation.
    The atlas template will serve as the moving image, and the annotation will be transformed.
    """
    print(f"[INFO] Fetching atlas information for: {atlas_name}")
    atlas = BrainGlobeAtlas(atlas_name)
    # The API returns atlas information with keys "template" and "annotation".
    atlas_reference = atlas.reference
    atlas_annotation = atlas.annotation
    print(atlas.orientation, atlas_annotation.shape, atlas_reference.shape)
    atlas_reference = ants.from_numpy(atlas_reference)
    atlas_annotation = ants.from_numpy(atlas_annotation)
    return atlas_reference, atlas_annotation


def register_atlas_to_downsampled(fixed_img, moving_img):
    """
    Register the moving image (atlas template) to the fixed image (downsampled image)
    using a non-linear registration (SyN).
    """
    print("[INFO] Registering atlas (moving) to downsampled image (fixed)...")
    reg_result = ants.registration(
        fixed=fixed_img, moving=moving_img, type_of_transform='SyN')
    return reg_result


def apply_transform_to_annotation(fixed_img, atlas_annotation, transforms):
    """
    Apply the computed transformation to the atlas annotation.
    Nearest-neighbor interpolation is used to preserve label values.
    """
    print("[INFO] Applying transforms to atlas annotation (using nearest-neighbor interpolation)...")
    transformed_annotation = ants.apply_transforms(
        fixed=fixed_img,
        moving=atlas_annotation,
        transformlist=transforms,
        interpolator='nearestNeighbor'
    )
    return transformed_annotation


def main():
    parser = argparse.ArgumentParser(
        description="Register BrainGlobe atlas to a downsampled Zarr image and transform the atlas annotation."
    )
    parser.add_argument("--downsampled_zarr", type=str, required=True,
                        help="Path to the downsampled image stored as a Zarr store.")
    parser.add_argument("--atlas_name", type=str, required=True,
                        help="BrainGlobe atlas name (e.g., allen_mouse_25um).")
    parser.add_argument("--output_annotation", type=str, required=True,
                        help="Output file path for the transformed atlas annotation (e.g., NIfTI file).")
    args = parser.parse_args()

    # Load the fixed image (downsampled image) from the Zarr store.
    fixed_img = load_downsampled_image_from_zarr(args.downsampled_zarr)

    # Load the atlas template and annotation using the BrainGlobe Atlas API.
    atlas_template, atlas_annotation = load_atlas_images(args.atlas_name)

    # Register the atlas template (moving) to the downsampled image (fixed).
    reg_result = register_atlas_to_downsampled(fixed_img, atlas_template)

    # Save transformation files to disk
    # Use directory of output_annotation or current dir
    output_dir = os.path.dirname(args.output_annotation) or "."
    output_prefix = os.path.basename(args.output_annotation).replace(
        # remove .zarr if present, to use as prefix
        ".zarr", "").replace('.nii.gz', '')
    transform_output_dir = os.path.join(
        output_dir, f"{output_prefix}_transforms")
    os.makedirs(transform_output_dir, exist_ok=True)
    print(f"[INFO] Saving transformation files to: {transform_output_dir}")

    for i, transform_path in enumerate(reg_result['fwdtransforms']):
        fwd_transform_output_path = os.path.join(
            transform_output_dir, f"fwd_transform_{i}.mat")
        shutil.copy(transform_path, fwd_transform_output_path)  # Copy the file
        print(f"[INFO] Saved forward transform: {fwd_transform_output_path}")
    for i, transform_path in enumerate(reg_result['invtransforms']):
        inv_transform_output_path = os.path.join(
            transform_output_dir, f"inv_transform_{i}.mat")
        shutil.copy(transform_path, inv_transform_output_path)  # Copy the file
        print(f"[INFO] Saved inverse transform: {inv_transform_output_path}")

    # Apply the computed transforms to the atlas annotation.
    transformed_annotation = apply_transform_to_annotation(
        fixed_img,
        atlas_annotation,
        reg_result['fwdtransforms']
    )

    ants.plot(fixed_img, transformed_annotation)

    # Save the transformed annotation to the specified output path.
    tiff.imwrite(args.output_annotation,
                 transformed_annotation.numpy().astype(np.uint16))
    # ants.image_write(transformed_annotation, args.output_annotation)
    print(
        f"[INFO] Transformed atlas annotation saved to: {args.output_annotation}")


if __name__ == '__main__':
    main()
