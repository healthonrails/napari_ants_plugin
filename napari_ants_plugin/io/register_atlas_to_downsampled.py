#!/usr/bin/env python3
"""
register_atlas_to_image.py

This script loads a fixed image (stored as a Zarr store or a TIF file),
fetches the atlas template and annotation via the BrainGlobe Atlas API using an atlas name (e.g., "allen_mouse_25um"),
registers the atlas (as the moving image) to the fixed image using ANTsPy with non-linear (SyN) registration,
applies the computed transform to the atlas annotation using nearest-neighbor interpolation,
and saves the transformed annotation to the specified output path.

Usage:
    python register_atlas_to_image.py \
      --fixed_image path/to/fixed_image.zarr_or_tif \
      --atlas_name allen_mouse_25um \
      --output_annotation transformed_annotation.nii.gz
"""

import argparse
import logging
import os

import dask.array as da
import numpy as np
import ants
import zarr
import tifffile as tiff
from brainglobe_atlasapi import BrainGlobeAtlas  # BrainGlobe Atlas API


def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def load_zarr_image(zarr_path: str) -> ants.ANTsImage:
    """
    Load an image from a Zarr store using Dask and convert it to an ANTs image.

    Parameters:
        zarr_path (str): Path to the Zarr store.

    Returns:
        ants.ANTsImage: The loaded image.
    """
    logging.info(f"Loading image from Zarr store: {zarr_path}")
    group = zarr.open(zarr_path, mode='r')
    # Assume the image data is stored under the key "data"
    zarr_array = group['data']
    darr = da.from_zarr(zarr_array)
    logging.info(f"Image shape from Zarr: {darr.shape}")
    np_img = darr.compute()
    return ants.from_numpy(np_img)


def load_tif_image(tif_path: str) -> ants.ANTsImage:
    """
    Load an image from a TIF file and convert it to an ANTs image.

    Parameters:
        tif_path (str): Path to the TIF file.

    Returns:
        ants.ANTsImage: The loaded image.
    """
    logging.info(f"Loading image from TIF file: {tif_path}")
    np_img = tiff.imread(tif_path)
    logging.info(f"Image shape from TIF: {np_img.shape}")
    return ants.from_numpy(np_img)


def load_fixed_image(image_path: str) -> ants.ANTsImage:
    """
    Load a fixed image from either a Zarr store or a TIF file based on the file extension.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        ants.ANTsImage: The loaded image.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    if ext == '.zarr':
        return load_zarr_image(image_path)
    elif ext in ['.tif', '.tiff']:
        return load_tif_image(image_path)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported formats are .zarr, .tif, and .tiff."
        )


def load_atlas_images(atlas_name: str) -> (ants.ANTsImage, ants.ANTsImage):
    """
    Retrieve the atlas template and annotation using the BrainGlobe Atlas API.

    Parameters:
        atlas_name (str): Name of the atlas (e.g., "allen_mouse_25um").

    Returns:
        tuple: (atlas_template, atlas_annotation) as ANTs images.
    """
    logging.info(f"Fetching atlas information for: {atlas_name}")
    atlas = BrainGlobeAtlas(atlas_name)
    atlas_reference = atlas.reference
    atlas_annotation = atlas.annotation
    logging.info(
        f"Atlas orientation: {atlas.orientation}, "
        f"template shape: {atlas_reference.shape}, "
        f"annotation shape: {atlas_annotation.shape}"
    )
    return ants.from_numpy(atlas_reference), ants.from_numpy(atlas_annotation)


def register_atlas_to_fixed(fixed_img: ants.ANTsImage,
                              moving_img: ants.ANTsImage) -> dict:
    """
    Register the atlas template (moving image) to the fixed image using non-linear (SyN) registration.

    Parameters:
        fixed_img (ants.ANTsImage): The fixed image.
        moving_img (ants.ANTsImage): The moving image (atlas template).

    Returns:
        dict: Registration results containing transformation parameters.
    """
    logging.info("Registering atlas (moving) to fixed image...")
    reg_result = ants.registration(fixed=fixed_img, moving=moving_img,
                                   type_of_transform='SyN')
    return reg_result


def apply_transform_to_annotation(fixed_img: ants.ANTsImage,
                                  atlas_annotation: ants.ANTsImage,
                                  transforms: list) -> ants.ANTsImage:
    """
    Apply the computed transforms to the atlas annotation using nearest-neighbor interpolation.

    Parameters:
        fixed_img (ants.ANTsImage): The fixed image.
        atlas_annotation (ants.ANTsImage): The atlas annotation to be transformed.
        transforms (list): List of forward transforms from registration.

    Returns:
        ants.ANTsImage: The transformed atlas annotation.
    """
    logging.info("Applying transforms to atlas annotation (using nearest-neighbor interpolation)...")
    transformed_annotation = ants.apply_transforms(
        fixed=fixed_img,
        moving=atlas_annotation,
        transformlist=transforms,
        interpolator='nearestNeighbor'
    )
    return transformed_annotation


def save_transformed_annotation(annotation: ants.ANTsImage,
                                output_path: str) -> None:
    """
    Save the transformed atlas annotation to the specified output path.

    Parameters:
        annotation (ants.ANTsImage): The transformed atlas annotation.
        output_path (str): File path to save the annotation.
    """
    # Convert annotation to uint16 before saving.
    annotation_array = annotation.numpy().astype(np.uint16)
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    if ext in ['.tif', '.tiff']:
        tiff.imwrite(output_path, annotation_array)
        logging.info(f"Transformed atlas annotation saved as TIF to: {output_path}")
    elif ext in ['.nii', '.nii.gz']:
        ants.image_write(annotation, output_path)
        logging.info(f"Transformed atlas annotation saved as NIfTI to: {output_path}")
    else:
        # Default to TIF if extension is unrecognized.
        tiff.imwrite(output_path, annotation_array)
        logging.info(f"Unrecognized extension; saved annotation as TIF to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Register BrainGlobe atlas to a fixed image (Zarr or TIF) and transform the atlas annotation."
    )
    parser.add_argument("--fixed_image", type=str, required=True,
                        help="Path to the fixed image stored as a Zarr store or TIF file.")
    parser.add_argument("--atlas_name", type=str, required=True,
                        help="BrainGlobe atlas name (e.g., allen_mouse_25um).")
    parser.add_argument("--output_annotation", type=str, required=True,
                        help="Output file path for the transformed atlas annotation (e.g., .nii.gz or .tif).")
    return parser.parse_args()


def main() -> None:
    """Main function to perform atlas registration and annotation transformation."""
    setup_logging()
    args = parse_arguments()

    try:
        fixed_img = load_fixed_image(args.fixed_image)
    except Exception as e:
        logging.error(f"Failed to load fixed image: {e}")
        return

    try:
        atlas_template, atlas_annotation = load_atlas_images(args.atlas_name)
    except Exception as e:
        logging.error(f"Failed to load atlas images: {e}")
        return

    reg_result = register_atlas_to_fixed(fixed_img, atlas_template)
    transformed_annotation = apply_transform_to_annotation(
        fixed_img,
        atlas_annotation,
        reg_result.get('fwdtransforms', [])
    )

    # Optionally, display the registration result.
    try:
        ants.plot(fixed_img, transformed_annotation)
    except Exception as e:
        logging.warning(f"Could not display plot: {e}")

    save_transformed_annotation(transformed_annotation, args.output_annotation)
    logging.info("Processing completed successfully.")


if __name__ == '__main__':
    main()
