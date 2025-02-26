import argparse
import logging
import os
import zarr
import dask.array as da
import gc
import time

from cellfinder.core.main import main as cellfinder_run
from cellfinder.core.classify import classify
from cellfinder.core.tools.prep import prep_models
from cellfinder.core.download.download import DEFAULT_DOWNLOAD_DIRECTORY
from brainglobe_utils.IO.cells import save_cells, cells_to_csv, get_cells

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
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=16,
        help="Batch size for the classification stage.",
    )
    parser.add_argument(
        "--classification-torch-device",
        type=str,
        default="cuda",
        help="Torch device for classification (e.g., 'cpu' or 'cuda').",
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


def run_detection(signal_array: da.Array, background_array: da.Array, voxel_sizes: list, n_free_cpus: int) -> list:
    """
    Run cell detection using the cellfinder_run method with classification skipped.

    Parameters:
        signal_array (da.Array): The signal data as a Dask array.
        background_array (da.Array): The background data as a Dask array.
        voxel_sizes (list): Voxel sizes (in microns) for detection.
        n_free_cpus (int): Number of CPUs to leave free.

    Returns:
        list: A list of detected cells.
    """
    return cellfinder_run(
        signal_array,
        background_array,
        voxel_sizes=voxel_sizes,
        n_free_cpus=n_free_cpus,
        skip_classification=True,
    )


def create_detected_cells_callback(logger):
    """
    Creates a callback function that logs progress information about detected cells.

    Parameters:
        logger (logging.Logger): Logger instance to log progress.

    Returns:
        function: A callback function that logs the current progress.
    """
    start_time = time.time()

    def detected_cells_callback(cells):
        elapsed = time.time() - start_time
        # Log the number of cells processed and the elapsed time.
        logger.info("Detection progress: %d cells detected after %.2f seconds.", len(
            cells), elapsed)
        return cells

    return detected_cells_callback


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
    try:
        logger.info("Preparing model weights...")
        model_weights = prep_models(
            None, DEFAULT_DOWNLOAD_DIRECTORY, args.model)
    except Exception as e:
        logger.error("Error preparing model weights: %s", str(e))
        return

    # Load signal and background arrays from Zarr files
    try:
        logger.info("Loading signal and background data...")
        signal_array = load_dask_array_from_zarr(args.signal)
        background_array = load_dask_array_from_zarr(args.background)
    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        return

    # Detection phase
    if os.path.exists(args.detected_cells_xml):
        logger.info("Skipping cell detection; file exists: %s",
                    args.detected_cells_xml)
        detected_cells = get_cells(args.detected_cells_xml)
    else:
        try:
            logger.info("Running cell detection...")
            detected_cells = run_detection(
                signal_array, background_array, args.voxel_sizes, args.n_free_cpus)
        except MemoryError as me:
            logger.error("Memory error during detection: %s", str(me))
            return
        except Exception as e:
            logger.error("Error during detection: %s", str(e))
            return

    # Save detected cells if output files do not already exist
    if not os.path.exists(args.detected_cells_xml):
        save_cells(detected_cells, args.detected_cells_xml)
        logger.info("Saved detected cells to %s.", args.detected_cells_xml)
    else:
        logger.info(
            "File %s already exists; skipping XML saving for detected cells.", args.detected_cells_xml)

    if not os.path.exists(args.detected_cells_csv):
        cells_to_csv(detected_cells, args.detected_cells_csv)
        logger.info("Saved detected cells to %s.", args.detected_cells_csv)
    else:
        logger.info(
            "File %s already exists; skipping CSV saving for detected cells.", args.detected_cells_csv)

    logger.info("Detected %d cells.", len(detected_cells))

    # Create a progress-monitoring callback for detection/classification.
    detected_cells_callback = create_detected_cells_callback(logger)

    # Classification phase
    if detected_cells:
        try:
            logger.info("Running classification on detected cells...")
            classified_cells = cellfinder_run(
                signal_array,
                background_array,
                voxel_sizes=args.voxel_sizes,
                detected_cells=detected_cells,
                classification_batch_size=args.classification_batch_size,
                skip_detection=True,
                classification_torch_device=args.classification_torch_device,
                n_free_cpus=args.n_free_cpus,
                model_weights=model_weights,
                batch_size=args.batch_size,
                trained_model=args.trained_model,
                network_depth=args.network_depth,
                cube_height=args.cube_height,
                cube_depth=args.cube_depth,
                cube_width=args.cube_width,
                network_voxel_sizes=args.network_voxel_sizes,
                detect_finished_callback=detected_cells_callback,
            )
        except MemoryError as me:
            logger.error("Memory error during classification: %s", str(me))
            return
        except Exception as e:
            logger.error("Error during classification: %s", str(e))
            return

        # Explicit memory cleanup after classification
        gc.collect()
        if args.classification_torch_device.lower().startswith("cuda"):
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        if not os.path.exists(args.classified_cells_xml):
            save_cells(classified_cells, args.classified_cells_xml)
            logger.info("Saved classified cells to %s.",
                        args.classified_cells_xml)
        else:
            logger.info(
                "File %s already exists; skipping XML saving for classified cells.", args.classified_cells_xml)

        if not os.path.exists(args.classified_cells_csv):
            cells_to_csv(classified_cells, args.classified_cells_csv)
            logger.info("Saved classified cells to %s.",
                        args.classified_cells_csv)
        else:
            logger.info(
                "File %s already exists; skipping CSV saving for classified cells.", args.classified_cells_csv)
    else:
        logger.info("No cells detected. Skipping classification.")


if __name__ == "__main__":
    main()
