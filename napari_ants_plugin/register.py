import ants
import numpy as np
import pandas as pd
import pathlib
import napari
import brainglobe_space as bg
from brainglobe_atlasapi import BrainGlobeAtlas
from brainreg.napari.util import downsample_and_save_brain
import tifffile as tiff
import argparse

# Constants and Configuration
POINTS_CSV_FILENAME = "points.csv"
TRANSFORMED_POINTS_CSV_FILENAME = "transformed_points.csv"
OUTPUT_DIR_NAME = "registration_output"
ANTS_REGISTRATION_PREFIX = "ants_reg_"
DEFAULT_TRANSFORM_TYPE = 'SyN'


def load_points_from_csv(csv_filename: str = POINTS_CSV_FILENAME) -> pd.DataFrame:
    """
    Loads points from a CSV file, assuming columns are in y, z, x order.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_filename}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_filename}: {e}")

    if not all(col in df.columns for col in ['y', 'z', 'x']):
        raise ValueError(
            f"CSV file {csv_filename} must contain columns 'y', 'z', 'x'."
        )
    return df[['y', 'z', 'x']]


def calculate_scaling_factors(image_voxel_sizes: tuple[float, float, float], atlas_key: str) -> tuple[float, float, float]:
    """
    Computes scaling factors to align image resolution with the atlas.
    """
    atlas = BrainGlobeAtlas(atlas_key)
    atlas_voxel = atlas.resolution

    if atlas_voxel is None:
        raise ValueError(
            f"Atlas '{atlas_key}' metadata lacks voxel_size information.")

    image_voxel = np.array(image_voxel_sizes, dtype=float)
    atlas_voxel = np.array(atlas_voxel, dtype=float)

    if not all(atlas_voxel > 0) or not all(image_voxel > 0):
        raise ValueError("Voxel sizes must be positive values.")

    scaling_factors = image_voxel / atlas_voxel  # Element-wise division
    return tuple(scaling_factors)


def load_atlas_and_orientation(atlas_key: str) -> tuple[BrainGlobeAtlas, str]:
    """
    Loads a BrainGlobe atlas and retrieves its orientation.
    """
    atlas = BrainGlobeAtlas(atlas_key)
    atlas_orientation = atlas.metadata.get("orientation")
    if atlas_orientation is None:
        raise ValueError(
            f"Atlas '{atlas_key}' metadata does not contain 'orientation'.")
    return atlas, atlas_orientation


def prepare_images_for_registration(
    img_layer: napari.layers.Image,
    atlas: BrainGlobeAtlas,
    input_voxel_sizes: tuple[float, float, float],
    input_orientation: str,
    atlas_orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares moving and fixed images for ANTs registration.
    """
    scaling = calculate_scaling_factors(input_voxel_sizes, atlas.atlas_name)
    print(f"Calculated scaling factors (z, y, x): {scaling}")

    # Downsample the brain image
    target_brain = downsample_and_save_brain(img_layer, scaling)
    # Reorient the input image (from its native orientation) to the atlas space
    reoriented_brain = bg.map_stack_to(
        input_orientation, atlas_orientation, target_brain)
    print(
        f"Input orientation: {input_orientation}, Atlas orientation: {atlas_orientation}")
    print(f"Reoriented brain shape: {reoriented_brain.shape}")

    fixed_np = atlas.reference.astype(np.float32)
    moving_np = reoriented_brain.astype(np.float32)
    print(f"Moving image shape for ANTs: {moving_np.shape}")
    return moving_np, fixed_np


def run_ants_registration(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    atlas: BrainGlobeAtlas,
    output_dir: pathlib.Path,
    transform_type: str = DEFAULT_TRANSFORM_TYPE,
) -> dict:
    """
    Executes ANTs registration between a moving and a fixed image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / ANTS_REGISTRATION_PREFIX)

    atlas_voxel = atlas.resolution
    fixed = ants.from_numpy(fixed_image, spacing=atlas_voxel)
    moving = ants.from_numpy(moving_image, spacing=atlas_voxel)

    print(f"Running ANTs registration with transform type: {transform_type}")
    reg_result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        outprefix=output_prefix,
        verbose=True
    )
    print("ANTs Registration complete.")
    return reg_result


def apply_transforms_and_save(
    reg_result: dict,
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    output_dir: pathlib.Path,
    atlas_orientation: str,
    target_orientation: str = "asr"
) -> dict:
    """
    Applies forward and inverse transforms from ANTs registration,
    reorients the downsampled registered image to target_orientation,
    and saves/returns results.
    """
    fwd_transforms = reg_result['fwdtransforms']
    inv_transforms = reg_result['invtransforms']

    # Save the warped (registered) moving image
    downsampled_registered = reg_result["warpedmovout"].numpy()

    # Reorient from atlas orientation back to the desired target orientation (e.g. "asr")
    if atlas_orientation != target_orientation:
        print(
            f"Reorienting the registered image from {atlas_orientation} to {target_orientation}.")
        downsampled_registered = bg.map_stack_to(
            atlas_orientation, target_orientation, downsampled_registered)
    else:
        print("No reorientation needed; atlas orientation matches target orientation.")

    tiff.imwrite(str(output_dir.parent / "downsampled_atlas.tif"),
                 downsampled_registered)

    # Example: applying the inverse transforms to get an annotation image
    fixed_ants = ants.from_numpy(fixed_image)
    moving_ants = ants.from_numpy(moving_image)
    moved_backward_ants = ants.apply_transforms(
        fixed=moving_ants,
        moving=fixed_ants,
        transformlist=inv_transforms,
        interpolator="nearestNeighbor",
    )

    results = {
        "transformed_moving": downsampled_registered,
        "annotation": moved_backward_ants.numpy(),
        "fwd_transform_files": fwd_transforms,
        "inv_transform_files": inv_transforms,
    }
    print("Transforms applied and results prepared.")
    print(f"Forward transforms: {fwd_transforms}")
    print(f"Inverse transforms: {inv_transforms}")
    return results


def run_registration_pipeline_ants(
    img_layer: napari.layers.Image,
    atlas_key: str,
    input_voxel_sizes: tuple[float, float, float],
    input_orientation: str,
    output_dir: str,
    additional_params: dict | None = None,
) -> dict:
    """
    Executes the complete registration pipeline using ANTs.
    """
    print(f"Starting ANTs registration pipeline for atlas: {atlas_key}")

    output_path = pathlib.Path(output_dir)
    atlas, atlas_orientation = load_atlas_and_orientation(atlas_key)
    moving_image_np, fixed_image_np = prepare_images_for_registration(
        img_layer, atlas, input_voxel_sizes, input_orientation, atlas_orientation
    )

    # Allow overriding of the transform type via additional_params
    transform_type = additional_params.get(
        "transform", DEFAULT_TRANSFORM_TYPE) if additional_params else DEFAULT_TRANSFORM_TYPE

    reg_result = run_ants_registration(
        moving_image_np, fixed_image_np, atlas, output_path, transform_type=transform_type
    )
    results = apply_transforms_and_save(
        reg_result, moving_image_np, atlas.annotation, output_path,
        atlas_orientation, target_orientation="asr"
    )

    print(
        f"ANTs registration pipeline completed. Results saved to: {output_dir}")
    return results


def main():
    # Set up argparse so that users can run the program with command-line arguments
    parser = argparse.ArgumentParser(
        description="Brain Registration Pipeline using ANTs and Napari."
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input TIF image (e.g., 'MF1wt_125F_W_BS_488_downsampled_gpu.tif').")
    parser.add_argument("--points", type=str, default="points.csv",
                        help="Path to the points CSV file. Default is 'points.csv'.")
    parser.add_argument("--atlas", type=str, required=True,
                        help="BrainGlobe atlas key (e.g., 'kim_mouse_25um').")
    parser.add_argument("--voxel-sizes", type=str, default="25,25,25",
                        help="Comma-separated voxel sizes (z,y,x) in microns. Default is '25,25,25'.")
    parser.add_argument("--orientation", type=str, default="asr",
                        help="Input orientation code (default: 'asr').")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_NAME,
                        help=f"Directory to save registration outputs (default: '{OUTPUT_DIR_NAME}').")
    parser.add_argument("--transform", type=str, default=DEFAULT_TRANSFORM_TYPE,
                        help=f"Type of ANTs transform (default: '{DEFAULT_TRANSFORM_TYPE}').")
    args = parser.parse_args()

    # Parse voxel sizes from comma-separated string to tuple of floats
    try:
        voxel_sizes = tuple(map(float, args.voxel_sizes.split(',')))
        if len(voxel_sizes) != 3:
            raise ValueError
    except ValueError:
        raise ValueError(
            "Please provide three comma-separated values for voxel sizes (z,y,x).")

    # Create a napari viewer
    viewer = napari.Viewer()

    # Load image and add to viewer
    image_data = tiff.imread(args.image)
    img_layer = viewer.add_image(image_data, name="Brain")

    # Load points CSV (if any)
    points_xyz_initial = load_points_from_csv(args.points)

    # Run the registration pipeline with parameters from argparse
    results = run_registration_pipeline_ants(
        img_layer,
        args.atlas,
        voxel_sizes,
        args.orientation,
        args.output,
        additional_params={"transform": args.transform}
    )

    # Add the registered (and now reoriented to "asr") image to the viewer
    viewer.add_image(results["transformed_moving"], name="Transformed")

    # Optionally, transform the points using the forward transforms (if present)
    if results["fwd_transform_files"]:
        transformed_points = ants.apply_transforms_to_points(
            dim=3,
            points=points_xyz_initial,
            transformlist=results["fwd_transform_files"],
        )

        df_orig = pd.DataFrame(points_xyz_initial, columns=[
                               "original_x", "original_y", "original_z"])
        df_trans = pd.DataFrame(transformed_points, columns=[
                                "transformed_x", "transformed_y", "transformed_z"])
        df_combined = pd.concat([df_orig, df_trans], axis=1)
        df_combined.to_csv(TRANSFORMED_POINTS_CSV_FILENAME, index=False)
        print(f"Saved combined points to '{TRANSFORMED_POINTS_CSV_FILENAME}'.")
    else:
        print("No forward transforms found; skipping point transformation.")

    napari.run()


if __name__ == "__main__":
    main()
