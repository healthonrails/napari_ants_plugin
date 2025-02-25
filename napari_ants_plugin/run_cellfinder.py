import argparse
import logging
import os
import zarr
import dask.array as da

from cellfinder.core.main import main as cellfinder_run
from cellfinder.core.classify import classify
from cellfinder.core.tools.prep import prep_models
from cellfinder.core.download.download import DEFAULT_DOWNLOAD_DIRECTORY
from brainglobe_utils.IO.cells import save_cells, cells_to_csv

# =============================================================================
# Default Configuration Constants
# =============================================================================
DEFAULT_MODEL = "resnet50_tv"
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_FREE_CPUS = 1
DEFAULT_NETWORK_VOXEL_SIZES = [5, 1, 1]
DEFAULT_CUBE_WIDTH = 50
DEFAULT_CUBE_HEIGHT = 50
DEFAULT_CUBE_DEPTH = 20
DEFAULT_NETWORK_DEPTH = "50"
DEFAULT_VOXEL_SIZES = [4, 2, 2]  # in microns

DEFAULT_SIGNAL_ZARR_PATH = "output_pipeline/intermediate/MF1_126F_W_BS_640.zarr"
DEFAULT_BACKGROUND_ZARR_PATH = "output_pipeline/intermediate/MF1_126F_W_BS_488.zarr"

# Default output file names for cell data
DEFAULT_DETECTED_CELLS_XML = "detected_cells_cellfinder.xml"
DEFAULT_DETECTED_CELLS_CSV = "detected_cells_cellfinder.csv"
DEFAULT_CLASSIFIED_CELLS_XML = "classified_cells_cellfinder.xml"
DEFAULT_CLASSIFIED_CELLS_CSV = "classified_cells_cellfinder.csv"


# =============================================================================
# Helper Functions
# =============================================================================
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Cell detection and classification using Cellfinder."
    )
    parser.add_argument(
        "--signal",
        type=str,
        default=DEFAULT_SIGNAL_ZARR_PATH,
        help="Path to the signal Zarr file.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default=DEFAULT_BACKGROUND_ZARR_PATH,
        help="Path to the background Zarr file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name to use for classification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for classification.",
    )
    parser.add_argument(
        "--n-free-cpus",
        type=int,
        default=DEFAULT_N_FREE_CPUS,
        help="Number of free CPUs to use.",
    )
    parser.add_argument(
        "--network-voxel-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_NETWORK_VOXEL_SIZES,
        help="List of network voxel sizes.",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_VOXEL_SIZES,
        help="List of voxel sizes (in microns) for detection.",
    )
    parser.add_argument(
        "--cube-width",
        type=int,
        default=DEFAULT_CUBE_WIDTH,
        help="Width of the cube used in classification.",
    )
    parser.add_argument(
        "--cube-height",
        type=int,
        default=DEFAULT_CUBE_HEIGHT,
        help="Height of the cube used in classification.",
    )
    parser.add_argument(
        "--cube-depth",
        type=int,
        default=DEFAULT_CUBE_DEPTH,
        help="Depth of the cube used in classification.",
    )
    parser.add_argument(
        "--network-depth",
        type=str,
        default=DEFAULT_NETWORK_DEPTH,
        help="Network depth configuration.",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=None,
        help="Path to a pre-trained model (optional).",
    )
    # New arguments for cell-related output files
    parser.add_argument(
        "--detected-cells-xml",
        type=str,
        default=DEFAULT_DETECTED_CELLS_XML,
        help="File path for saving detected cells (XML). If the file exists, saving will be skipped.",
    )
    parser.add_argument(
        "--detected-cells-csv",
        type=str,
        default=DEFAULT_DETECTED_CELLS_CSV,
        help="File path for saving detected cells (CSV). If the file exists, saving will be skipped.",
    )
    parser.add_argument(
        "--classified-cells-xml",
        type=str,
        default=DEFAULT_CLASSIFIED_CELLS_XML,
        help="File path for saving classified cells (XML). If the file exists, saving will be skipped.",
    )
    parser.add_argument(
        "--classified-cells-csv",
        type=str,
        default=DEFAULT_CLASSIFIED_CELLS_CSV,
        help="File path for saving classified cells (CSV). If the file exists, saving will be skipped.",
    )
    return parser.parse_args()


def load_dask_array_from_zarr(zarr_path: str, dataset_key: str = "data") -> da.Array:
    """
    Load a Dask array from a Zarr file.

    Parameters:
        zarr_path (str): Path to the Zarr file.
        dataset_key (str): Key for the dataset inside the Zarr container.

    Returns:
        da.Array: A Dask array representing the dataset.
    """
    z = zarr.open(zarr_path, mode="r")
    return da.from_zarr(z[dataset_key])


def run_detection(signal_array: da.Array, background_array: da.Array, voxel_sizes: list) -> list:
    """
    Run cell detection using the cellfinder_run method.

    Parameters:
        signal_array (da.Array): The signal data as a Dask array.
        background_array (da.Array): The background data as a Dask array.
        voxel_sizes (list): Voxel sizes (in microns) for detection.

    Returns:
        list: A list of detected cells.
    """
    return cellfinder_run(signal_array, background_array,
                          batch_size=8,
                          n_free_cpus=2,
                          classification_batch_size=8,
                          classification_torch_device='cuda',
                          skip_classification=True,
                          )


def run_classification(
    detected_cells: list,
    signal_array: da.Array,
    background_array: da.Array,
    n_free_cpus: int,
    voxel_sizes: list,
    network_voxel_sizes: list,
    batch_size: int,
    cube_height: int,
    cube_width: int,
    cube_depth: int,
    trained_model,
    model_weights,
    network_depth: str,
) -> list:
    """
    Run cell classification if there are any detected cells.

    Parameters:
        detected_cells (list): List of cells detected.
        signal_array (da.Array): The signal data as a Dask array.
        background_array (da.Array): The background data as a Dask array.
        n_free_cpus (int): Number of free CPUs.
        voxel_sizes (list): Voxel sizes for detection.
        network_voxel_sizes (list): Network voxel sizes for classification.
        batch_size (int): Batch size for classification.
        cube_height (int): Cube height used in classification.
        cube_width (int): Cube width used in classification.
        cube_depth (int): Cube depth used in classification.
        trained_model: Pre-trained model (optional).
        model_weights: Prepared model weights for classification.
        network_depth (str): Network depth configuration.

    Returns:
        list: A list of classified cells.
    """
    return classify.main(
        detected_cells,
        signal_array,
        background_array,
        n_free_cpus,
        voxel_sizes,
        network_voxel_sizes,
        batch_size,
        cube_height,
        cube_width,
        cube_depth,
        trained_model,
        model_weights,
        network_depth,
    )


# =============================================================================
# Main Execution Flow
# =============================================================================
def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_args()

    # Prepare model weights
    logger.info("Preparing model weights...")
    model_weights = prep_models(None, DEFAULT_DOWNLOAD_DIRECTORY, args.model)

    # Load signal and background arrays from Zarr files
    logger.info("Loading signal and background data...")
    signal_array = load_dask_array_from_zarr(args.signal)
    background_array = load_dask_array_from_zarr(args.background)

    # Run cell detection
    logger.info("Running cell detection...")
    detected_cells = run_detection(
        signal_array, background_array, args.voxel_sizes)

    # Save detected cells if output files do not already exist
    if not os.path.exists(args.detected_cells_xml):
        save_cells(detected_cells, args.detected_cells_xml)
        logger.info(f"Saved detected cells to {args.detected_cells_xml}.")
    else:
        logger.info(
            f"File {args.detected_cells_xml} already exists; skipping XML saving.")

    if not os.path.exists(args.detected_cells_csv):
        cells_to_csv(detected_cells, args.detected_cells_csv)
        logger.info(f"Saved detected cells to {args.detected_cells_csv}.")
    else:
        logger.info(
            f"File {args.detected_cells_csv} already exists; skipping CSV saving.")

    logger.info(f"Detected {len(detected_cells)} cells.")

    # Run classification if cells are detected
    if detected_cells:
        logger.info("Detected cells found. Running classification...")
        classified_cells = run_classification(
            detected_cells,
            signal_array,
            background_array,
            args.n_free_cpus,
            args.voxel_sizes,
            args.network_voxel_sizes,
            args.batch_size,
            args.cube_height,
            args.cube_width,
            args.cube_depth,
            args.trained_model,
            model_weights,
            args.network_depth,
        )

        # Save classified cells if output files do not already exist
        if not os.path.exists(args.classified_cells_xml):
            save_cells(classified_cells, args.classified_cells_xml)
            logger.info(
                f"Saved classified cells to {args.classified_cells_xml}.")
        else:
            logger.info(
                f"File {args.classified_cells_xml} already exists; skipping XML saving.")

        if not os.path.exists(args.classified_cells_csv):
            cells_to_csv(classified_cells, args.classified_cells_csv)
            logger.info(
                f"Saved classified cells to {args.classified_cells_csv}.")
        else:
            logger.info(
                f"File {args.classified_cells_csv} already exists; skipping CSV saving.")
    else:
        logger.info("No cells detected. Skipping classification.")


if __name__ == "__main__":
    main()
