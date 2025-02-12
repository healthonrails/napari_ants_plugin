#!/usr/bin/env python3
"""
Brain Registration Pipeline using ANTs and Napari.

This script registers a brain image to an atlas using the ANTs registration toolkit.
It also loads points from a CSV file and applies the computed transforms to the points.
Before applying the ANTs forward transforms, the original (e.g. “ial”) points are
reoriented into the atlas space and then (if necessary) into the target (“asr”) orientation.
The combined (original and transformed) points are then saved to a CSV file that is
Napari‐compatible.
"""

import argparse
import logging
import pathlib
import re
from typing import Any, Dict, Optional, Tuple

import ants
import brainglobe_space as bg
import napari
import numpy as np
import pandas as pd
import tifffile as tiff
from brainglobe_atlasapi import BrainGlobeAtlas
from brainreg.napari.util import downsample_and_save_brain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRANSFORM_TYPE = "SyN"
POINTS_CSV_FILENAME = "points.csv"
TRANSFORMED_POINTS_CSV_FILENAME = "transformed_points.csv"
OUTPUT_DIR_NAME = "registration_output"
ANTS_REGISTRATION_PREFIX = "ants_reg_"
LOG_FILENAME = "downsample_gpu.log"


class BrainRegistrationPipeline:
    def __init__(
        self,
        image_path: pathlib.Path,
        atlas_key: str,
        voxel_sizes: Tuple[float, float, float],
        input_orientation: str,
        output_dir: pathlib.Path,
        transform_type: str = DEFAULT_TRANSFORM_TYPE,
        points_csv: Optional[pathlib.Path] = None,
        log_filename: str = LOG_FILENAME,
    ):
        self.image_path = image_path
        self.atlas_key = atlas_key
        self.voxel_sizes = voxel_sizes
        self.input_orientation = input_orientation
        self.output_dir = output_dir
        self.transform_type = transform_type
        self.points_csv = points_csv
        self.log_filename = log_filename

        # Load atlas and its orientation
        self.atlas, self.atlas_orientation = load_atlas_and_orientation(
            atlas_key)

    def run(self) -> Dict[str, Any]:
        """Executes the complete registration pipeline."""
        logger.info("Starting registration pipeline for atlas: %s",
                    self.atlas_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load image data
        image_data = tiff.imread(str(self.image_path))
        logger.info("Loaded image '%s' with shape %s",
                    self.image_path, image_data.shape)

        # Create a Napari viewer and add the image layer
        viewer = napari.Viewer()
        img_layer = viewer.add_image(image_data, name="Brain")

        # Prepare moving and fixed images
        moving_np, fixed_np = prepare_images_for_registration(
            img_layer,
            self.atlas,
            self.voxel_sizes,
            self.input_orientation,
            self.atlas_orientation,
        )

        # Check for existing registration output
        registration_output_file = self.output_dir.parent / "downsampled_atlas.tif"
        if registration_output_file.exists():
            logger.info(
                "Registration output '%s' exists; skipping registration.", registration_output_file
            )
            transformed_moving = tiff.imread(str(registration_output_file))
            results = {
                "transformed_moving": transformed_moving,
                "annotation": None,
                "fwd_transform_files": self._get_expected_transform_files(),
                "inv_transform_files": None,
                "transformed_points": None,
            }
        else:
            # Run ANTs registration
            reg_result = run_ants_registration(
                moving_np, fixed_np, self.atlas, self.output_dir, transform_type=self.transform_type
            )
            results = apply_transforms_and_save(
                reg_result,
                moving_np,
                self.atlas.annotation,
                self.output_dir,
                self.atlas_orientation,
                target_orientation="asr",
                atlas_annotation=self.atlas.annotation,
                cell_points=None,  # Will be updated if point data is provided
            )
            logger.info(
                "Registration completed; results saved to '%s'", self.output_dir)

        # Process points if a valid CSV file is provided
        if self.points_csv and self.points_csv.exists():
            points_df = process_points_csv(
                self.points_csv,
                self.log_filename,
                self.atlas,
                self.atlas_orientation,
                source_orientation="ial",
            )
            # If registration produced transformed points, use them; otherwise, use processed points
            if results.get("transformed_points") is not None:
                transformed_points = results["transformed_points"]
                points_transformed_df = pd.DataFrame(
                    transformed_points, columns=["x", "y", "z"])
            else:
                points_transformed_df = points_df.copy()

            # For each transformed point, query the atlas for structure info.
            points_transformed_df["structure"] = points_transformed_df.apply(
                lambda row: safe_structure_from_coords(
                    self.atlas,
                    (int(round(row["x"])), int(
                        round(row["y"])), int(round(row["z"]))),
                    as_acronym=True,
                ),
                axis=1,
            )
            # Combine original and transformed points
            original_points_df = points_df.rename(
                columns={"y": "original_y",
                         "z": "original_z", "x": "original_x"}
            )
            combined_df = pd.concat(
                [original_points_df, points_transformed_df], axis=1)
            # Remove points that are out of the brain region
            combined_df = combined_df[combined_df["structure"]
                                      != "out_of_brain_region"]
            combined_df.to_csv(TRANSFORMED_POINTS_CSV_FILENAME, index=False)
            logger.info("Saved transformed points to '%s'.",
                        TRANSFORMED_POINTS_CSV_FILENAME)
            viewer.add_points(
                combined_df[["z", "y", "x"]].values, size=5, face_color="red", name="Cells")
        else:
            logger.info(
                "No valid points CSV provided; skipping point processing.")

        napari.run()
        return results

    def _get_expected_transform_files(self) -> list:
        """Return the expected forward transform file paths."""
        return [
            self.output_dir / f"{ANTS_REGISTRATION_PREFIX}0GenericAffine.mat",
            self.output_dir / f"{ANTS_REGISTRATION_PREFIX}1Warp.nii.gz",
        ]


def extract_scaling_factors(log_filename: str = LOG_FILENAME) -> Tuple[float, float, float]:
    """
    Extract scaling factors from the log file.

    The log file should contain a line like:
      "Scaling factors - Depth: 0.16, Height: 0.06, Width: 0.08"

    Returns:
        A tuple (z_factor, y_factor, x_factor).
    """
    log_path = pathlib.Path(log_filename)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file '{log_filename}' does not exist.")

    log_text = log_path.read_text()
    pattern = r"Scaling factors\s*-\s*Depth:\s*([\d\.]+),\s*Height:\s*([\d\.]+),\s*Width:\s*([\d\.]+)"
    match = re.search(pattern, log_text)
    if not match:
        raise ValueError(
            f"Scaling factors not found in log file '{log_filename}'.")

    z_factor = float(match.group(1))
    y_factor = float(match.group(2))
    x_factor = float(match.group(3))
    return z_factor, y_factor, x_factor


def load_points_from_csv(csv_filename: str = POINTS_CSV_FILENAME) -> pd.DataFrame:
    """
    Load points from a CSV file assuming columns are 'z', 'y', 'x'.

    Returns:
        DataFrame containing the required columns.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", csv_filename)
        raise
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_filename}: {e}")

    required_columns = ["z", "y", "x"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV file {csv_filename} must contain columns {required_columns}")

    return df[required_columns]


def calculate_scaling_factors(
    image_voxel_sizes: Tuple[float, float, float], atlas_key: str
) -> Tuple[float, float, float]:
    """
    Compute scaling factors to align image resolution with atlas resolution.

    Returns:
        A tuple (z, y, x) scaling factors.
    """
    atlas = BrainGlobeAtlas(atlas_key)
    atlas_voxel = atlas.resolution
    if atlas_voxel is None:
        raise ValueError(
            f"Atlas '{atlas_key}' metadata lacks voxel_size information.")

    image_voxel = np.array(image_voxel_sizes, dtype=float)
    atlas_voxel = np.array(atlas_voxel, dtype=float)
    if not np.all(atlas_voxel > 0) or not np.all(image_voxel > 0):
        raise ValueError("Voxel sizes must be positive values.")

    scaling_factors = image_voxel / atlas_voxel
    logger.info("Calculated scaling factors (z, y, x): %s", scaling_factors)
    return tuple(scaling_factors)


def load_atlas_and_orientation(atlas_key: str) -> Tuple[BrainGlobeAtlas, str]:
    """
    Load a BrainGlobe atlas and return it along with its orientation.

    Returns:
        Tuple (atlas, atlas_orientation).
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
    input_voxel_sizes: Tuple[float, float, float],
    input_orientation: str,
    atlas_orientation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare moving and fixed images for registration by downsampling and reorienting.

    Returns:
        A tuple (moving_image, fixed_image) as numpy arrays.
    """
    scaling = calculate_scaling_factors(input_voxel_sizes, atlas.atlas_name)
    target_brain = downsample_and_save_brain(img_layer, scaling)
    reoriented_brain = bg.map_stack_to(
        input_orientation, atlas_orientation, target_brain)
    logger.info("Reoriented brain shape: %s", reoriented_brain.shape)

    fixed_np = atlas.reference.astype(np.float32)
    moving_np = reoriented_brain.astype(np.float32)
    logger.info("Moving image shape for ANTs: %s", moving_np.shape)
    return moving_np, fixed_np


def run_ants_registration(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    atlas: BrainGlobeAtlas,
    output_dir: pathlib.Path,
    transform_type: str = DEFAULT_TRANSFORM_TYPE,
) -> Dict[str, Any]:
    """
    Run ANTs registration between the moving and fixed images.

    Returns:
        Dictionary containing registration results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / ANTS_REGISTRATION_PREFIX)

    fixed = ants.from_numpy(fixed_image, spacing=atlas.resolution)
    moving = ants.from_numpy(moving_image, spacing=atlas.resolution)

    logger.info(
        "Running ANTs registration with transform type: %s", transform_type)
    reg_result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        outprefix=output_prefix,
        verbose=True,
    )
    logger.info("ANTs registration complete.")
    return reg_result


def apply_transforms_and_save(
    reg_result: Dict[str, Any],
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    output_dir: pathlib.Path,
    atlas_orientation: str,
    target_orientation: str = "asr",
    atlas_annotation: Any = None,
    cell_points: Any = None,
) -> Dict[str, Any]:
    """
    Apply computed transforms, reorient the registered image, and save outputs.

    Returns:
        Dictionary with transformed image, annotation image, and transform files.
    """
    fwd_transforms = reg_result.get("fwdtransforms", [])
    inv_transforms = reg_result.get("invtransforms", [])

    # Get warped moving image from registration results
    registered_image = reg_result["warpedmovout"].numpy()

    if atlas_orientation != target_orientation:
        logger.info("Reorienting registered image from %s to %s",
                    atlas_orientation, target_orientation)
        registered_image = bg.map_stack_to(
            atlas_orientation, target_orientation, registered_image)
    else:
        logger.info("No reorientation needed; orientations match.")

    output_image_path = output_dir.parent / "downsampled_atlas.tif"
    tiff.imwrite(str(output_image_path), registered_image)
    logger.info("Saved registered image to '%s'.", output_image_path)

    annotation_image = None
    if atlas_annotation is not None:
        fixed_ants = ants.from_numpy(moving_image)
        moving_ants = ants.from_numpy(atlas_annotation)
        moved_backward = ants.apply_transforms(
            fixed=moving_ants,
            moving=fixed_ants,
            transformlist=inv_transforms,
            interpolator="nearestNeighbor",
        )
        annotation_path = output_dir.parent / "annotation_warped.tif"
        tiff.imwrite(str(annotation_path), moved_backward.numpy())
        annotation_image = moved_backward.numpy()
        logger.info("Saved atlas annotation image to '%s'.", annotation_path)

    transformed_points = None
    if cell_points is not None:
        transformed_points = ants.apply_transforms_to_points(
            dim=3, points=cell_points, transformlist=fwd_transforms, verbose=True
        )

    results = {
        "transformed_moving": registered_image,
        "annotation": annotation_image,
        "fwd_transform_files": fwd_transforms,
        "inv_transform_files": inv_transforms,
        "transformed_points": transformed_points,
    }
    logger.info("Applied transforms and prepared results.")
    return results


def safe_structure_from_coords(
    atlas: BrainGlobeAtlas,
    coords: Tuple[int, int, int],
    default: str = "out_of_brain_region",
    as_acronym: bool = False,
) -> str:
    """
    Query the atlas for structure info at given coordinates.

    Returns:
        Structure acronym or a default value if the coordinate is invalid.
    """
    try:
        return atlas.structure_from_coords(coords, as_acronym=as_acronym)
    except IndexError:
        return default


def process_points_csv(
    points_csv: pathlib.Path,
    log_filename: str,
    atlas: BrainGlobeAtlas,
    atlas_orientation: str,
    source_orientation: str = "ial",
) -> pd.DataFrame:
    """
    Load, scale, and reorient points from a CSV file.

    Returns:
        DataFrame of reoriented points with columns ['z', 'y', 'x'].
    """
    points_df = load_points_from_csv(str(points_csv))
    original_points = points_df.copy()

    z_factor, y_factor, x_factor = extract_scaling_factors(log_filename)
    points_df["z"] = points_df["z"] * z_factor
    points_df["y"] = points_df["y"] * y_factor
    points_df["x"] = points_df["x"] * x_factor
    logger.info("Scaled points with factors: z=%s, y=%s, x=%s",
                z_factor, y_factor, x_factor)

    source_space = bg.AnatomicalSpace(
        source_orientation, atlas.reference.shape)
    target_space = bg.AnatomicalSpace(atlas_orientation, atlas.reference.shape)
    mapped_points = source_space.map_points_to(target_space, points_df)
    mapped_df = pd.DataFrame(mapped_points, columns=["z", "y", "x"])
    logger.info("Mapped points to atlas orientation '%s'.", atlas_orientation)
    return mapped_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain Registration Pipeline using ANTs and Napari."
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input TIF image.")
    parser.add_argument(
        "--points",
        type=str,
        default=POINTS_CSV_FILENAME,
        help="Path to the points CSV file (default: 'points.csv').",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        required=True,
        help="BrainGlobe atlas key (e.g., 'kim_mouse_25um').",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=str,
        default="25,25,25",
        help="Comma-separated voxel sizes (z,y,x) in microns (default: '25,25,25').",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="asr",
        help="Input orientation code (default: 'asr'). Use 'ial' if your CSV is in that orientation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR_NAME,
        help=f"Directory to save outputs (default: '{OUTPUT_DIR_NAME}').",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=DEFAULT_TRANSFORM_TYPE,
        help=f"Type of ANTs transform (default: '{DEFAULT_TRANSFORM_TYPE}').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        voxel_sizes = tuple(map(float, args.voxel_sizes.split(",")))
        if len(voxel_sizes) != 3:
            raise ValueError(
                "Provide three comma-separated values for voxel sizes (z,y,x).")
    except ValueError as e:
        logger.error("Error parsing voxel sizes: %s", e)
        raise

    pipeline = BrainRegistrationPipeline(
        image_path=pathlib.Path(args.image),
        atlas_key=args.atlas,
        voxel_sizes=voxel_sizes,
        input_orientation=args.orientation,
        output_dir=pathlib.Path(args.output),
        transform_type=args.transform,
        points_csv=pathlib.Path(args.points) if args.points else None,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
