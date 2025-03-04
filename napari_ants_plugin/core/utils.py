import os
import ants
import logging
from skimage.io import imsave
import datetime
import pandas as pd

logger = logging.getLogger(__name__)  # Get the logger
logger.setLevel(logging.INFO)  # Set the logging level


def annotate_cells_with_regions(detected_cells, annotation_zarr_path, atlas_name, output_csv_path):
    """
    Annotate a list of detected cells with brain region names based on an annotation volume,
    and save the results to a CSV file.

    Parameters:
      detected_cells (list): List of Cell objects. Each cell is expected to have attributes 
                             'x', 'y', and 'z' (and optionally 'type').
      annotation_zarr_path (str): Path to the Zarr store containing the annotation volume (under key "data").
      atlas_name (str): Name of the atlas to use for region lookup.
      output_csv_path (str): File path where the annotated cell data CSV will be saved.
    """
    import zarr
    from brainglobe_atlasapi import BrainGlobeAtlas
    from napari_ants_plugin.regions_gpu import add_region_info_to_points
    # Convert the list of Cell objects into a DataFrame.
    # Adjust the fields as needed.
    cell_data = []
    for cell in detected_cells:
        cell_data.append({
            "x": cell.x,
            "y": cell.y,
            "z": cell.z,
            "cell_type": cell.type  # Optional; include if desired.
        })
    df_cells = pd.DataFrame(cell_data)

    if df_cells.empty:
        print("No cells to annotate.")
        return

    # Load the annotation volume from the Zarr store.
    anno_z = zarr.open(annotation_zarr_path, mode="r")
    # Assumes annotation is stored under the key "data".
    # For large volumes consider lazy loading if needed.
    annotation = anno_z["data"]

    # Initialize the atlas instance.
    atlas = BrainGlobeAtlas(atlas_name)

    # Add region information to the DataFrame using your utility function.
    df_annotated = add_region_info_to_points(df_cells, annotation, atlas)

    # Save the annotated cell data to CSV.
    df_annotated.to_csv(output_csv_path, index=False)
    print(f"Annotated cells saved to {output_csv_path}")


def create_result_directory(result_dir: str) -> str:
    """Create or validate a result directory for saving transformations."""
    if not result_dir:
        result_dir = os.path.join(os.getcwd(), 'ANTs_results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def compose_transforms(transform_dir: str, invert: bool = False):
    """Compose a set of ANTs transforms from a directory.

    Parameters:
    -----------
    transform_dir : str
        Directory with ANTs transform files.
    invert : bool, optional
        Whether to invert the transformation.

    Returns:
    --------
    ants.ANTsTransform
        A composite transformation object.
    """
    transforms = []
    if not invert:
        if '1Warp.nii.gz' in os.listdir(transform_dir):
            SyN_file = os.path.join(transform_dir, '1Warp.nii.gz')
            field = ants.image_read(SyN_file)
            transform = ants.transform_from_displacement_field(field)
            transforms.append(transform)
        if '0GenericAffine.mat' in os.listdir(transform_dir):
            affine_file = os.path.join(transform_dir, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file))
    else:
        if '0GenericAffine.mat' in os.listdir(transform_dir):
            affine_file = os.path.join(transform_dir, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file).invert())
        if '1InverseWarp.nii.gz' in os.listdir(transform_dir):
            inv_file = os.path.join(transform_dir, '1InverseWarp.nii.gz')
            field = ants.image_read(inv_file)
            transform = ants.transform_from_displacement_field(field)
            transforms.append(transform)

    return ants.compose_ants_transforms(transforms)


def save_aligned_image(image, save_path: str) -> None:
    """Saves the aligned image, handling potential errors."""
    try:
        imsave(save_path, image)
        # Log successful save
        logger.info(f"Aligned image saved to: {save_path}")
        return save_path  # return the path for use elsewhere
    except Exception as e:  # Catch and log any exceptions during saving
        logger.error(f"Error saving aligned image: {e}")
        return None  # Indicate failure by returning None


def log_score(image_name: str, score: int, comments: str, log_dir: str = None) -> None:  # Add log_dir parameter
    """Logs the score and comments with timestamp and improved formatting."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if log_dir is None:
            log_dir = "."  # Current directory if log_dir is not provided
        log_path = os.path.join(
            log_dir, "registration_scores.log")  # .log extension

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "a") as f:
            f.write(
                f"{timestamp} - Image: {image_name}, Score: {score}, Comments: {comments}\n")
        logger.info(f"Score and comments logged to {log_path}")
    except Exception as e:
        logger.error(f"Error logging score: {e}")
