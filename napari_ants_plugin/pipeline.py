#!/usr/bin/env python3
"""
Mouse Brain Image Processing Pipeline

This pipeline performs the following steps:
  1. Convert background and signal TIFF images to Zarr.
  2. Downsample the background Zarr image (ensuring correct orientation).
  3. Register the atlas (moving image) to the downsampled background (fixed image).
  4. Upsample the transformed atlas annotation.
  5. Run CountGD-based cell labeling.
  6. Remove duplicate cell detections.
  7. Launch Napari viewer with sparse ROI extraction UI.

Each step is isolated as a method within the ImageProcessingPipeline class.
"""

import argparse
import logging
import os
import sys
from typing import Any

import pandas as pd

from napari_ants_plugin.io.tiff2zarr import convert_tiff_to_zarr
from napari_ants_plugin.io.downsample_gpu import downsample_zarr
from napari_ants_plugin.io.register_atlas_to_downsampled import main as run_registration
from napari_ants_plugin.io.upsample import AtlasUpsampler
from napari_ants_plugin.core.cells import remove_duplicate_cells


def setup_output_folders(base_dir: str) -> dict:
    """
    Create subfolders for logs, intermediate files, and final results.
    Returns a dictionary with the folder paths.
    """
    folders = {
        "base": base_dir,
        "logs": os.path.join(base_dir, "logs"),
        "intermediate": os.path.join(base_dir, "intermediate"),
        "results": os.path.join(base_dir, "results"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def setup_logger(log_file: str) -> logging.Logger:
    """
    Set up a logger that writes to both console and a log file.
    """
    logger = logging.getLogger("ImageProcessingPipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def check_file_exists(filepath: str) -> bool:
    """
    Return True if the file or directory exists and is non-empty.
    For Zarr stores (directories), check for the existence of the mandatory 'zarr.json' file.
    """
    if os.path.isdir(filepath):
        return os.path.exists(os.path.join(filepath, 'zarr.json'))
    else:
        return os.path.exists(filepath) and os.stat(filepath).st_size > 0


class ImageProcessingPipeline:
    """
    Encapsulates the end-to-end image processing pipeline, including launching
    the Napari viewer with sparse ROI extraction UI.
    """

    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.folders = setup_output_folders(self.config.output_dir)
        self.logger = logger
        self._setup_paths()

    def _setup_paths(self) -> None:
        """
        Compute all output file paths based on the output subdirectories.
        Filenames are derived from the input file names.
        """
        # Extract base names (without extension) from input files.
        bg_basename = os.path.splitext(
            os.path.basename(self.config.background))[0]
        signal_basename = os.path.splitext(
            os.path.basename(self.config.signal))[0]

        # Intermediate files.
        self.zarr_background_path = os.path.join(
            self.folders["intermediate"], f"{bg_basename}.zarr")
        self.zarr_signal_path = os.path.join(
            self.folders["intermediate"], f"{signal_basename}.zarr")
        self.downsampled_background_tif = os.path.join(
            self.folders["intermediate"], f"downsampled_{bg_basename}.tif")
        # Atlas-related file now includes atlas_name
        self.registered_annotation = os.path.join(
            self.folders["intermediate"],
            f"registered_{bg_basename}_{self.config.atlas_name}_annotation.tif"
        )

        # Final results.
        self.upsampled_annotation_zarr = os.path.join(
            self.folders["results"],
            f"upsampled_{bg_basename}_{self.config.atlas_name}_annotation.zarr"
        )
        # For viewer purposes, set unsampled_annotation_zarr equal to the upsampled annotation.
        self.unsampled_annotation_zarr = self.upsampled_annotation_zarr
        # For cell labeling outputs:
        self.detected_cells_csv = os.path.join(
            self.folders["results"], f"detected_cells_{signal_basename}.csv")
        self.unique_cells_csv = os.path.join(
            self.folders["results"], f"unique_cells_{signal_basename}.csv")
        # Atlas-related outputs: add atlas_name to file names.
        self.points_final_csv = os.path.join(
            self.folders["results"],
            f"points_final_{signal_basename}_{self.config.atlas_name}.csv"
        )
        self.cell_counts_csv = os.path.join(
            self.folders["results"],
            f"cell_counts_{signal_basename}_{self.config.atlas_name}.csv"
        )

    def run(self) -> None:
        """
        Execute the processing pipeline steps (steps 1-7).
        """
        self.logger.info("Pipeline started.")
        self._step_convert_background()
        self._step_convert_signal()
        self._step_downsample_background()
        self._step_register_atlas()
        self._step_upsample_atlas()
        self._step_cell_labeling()
        self._step_deduplicate_cells()
        self.logger.info("Pipeline completed successfully.")

    def _step_convert_background(self) -> None:
        if not check_file_exists(self.zarr_background_path):
            try:
                self.logger.info("Step 1: Converting background TIFF to Zarr")
                convert_tiff_to_zarr(self.config.background,
                                     self.zarr_background_path)
            except Exception as e:
                self.logger.error(f"Background conversion failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 1: Background Zarr already exists; skipping conversion.")

    def _step_convert_signal(self) -> None:
        if not check_file_exists(self.zarr_signal_path):
            try:
                self.logger.info("Step 2: Converting signal TIFF to Zarr")
                convert_tiff_to_zarr(self.config.signal, self.zarr_signal_path)
            except Exception as e:
                self.logger.error(f"Signal conversion failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 2: Signal Zarr already exists; skipping conversion.")

    def _step_downsample_background(self) -> None:
        if not check_file_exists(self.downsampled_background_tif):
            try:
                self.logger.info("Step 3: Downsampling background image")
                from brainglobe_atlasapi import BrainGlobeAtlas
                bg_atlas = BrainGlobeAtlas(self.config.atlas_name)
                downsample_zarr(
                    input_zarr_path=self.zarr_background_path,
                    output_tiff_path=self.downsampled_background_tif,
                    input_voxel_size=(4.0, 2.0, 2.0),
                    bg_atlas=bg_atlas,
                    logger=self.logger,
                    zarr_origin=self.config.input_orientation,
                )
            except Exception as e:
                self.logger.error(f"Downsampling failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 3: Downsampled background exists; skipping downsampling.")

    def _step_register_atlas(self) -> None:
        if not check_file_exists(self.registered_annotation):
            try:
                self.logger.info("Step 4: Registering atlas to background")
                reg_args = [
                    "--fixed_image", self.downsampled_background_tif,
                    "--atlas_name", self.config.atlas_name,
                    "--output_annotation", self.registered_annotation,
                    "--downsampled_orientation", self.config.orientation
                ]
                argv_backup = sys.argv.copy()
                sys.argv = [sys.argv[0]] + reg_args
                run_registration()
                sys.argv = argv_backup
            except Exception as e:
                self.logger.error(f"Atlas registration failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 4: Registered atlas exists; skipping registration.")

    def _step_upsample_atlas(self) -> None:
        if not check_file_exists(self.upsampled_annotation_zarr):
            try:
                self.logger.info("Step 5: Upsampling atlas annotation")
                upsampler = AtlasUpsampler(
                    annotation_file=self.registered_annotation,
                    final_zarr_path=self.upsampled_annotation_zarr,
                    atlas_voxel_size=tuple(
                        map(float, self.config.atlas_voxel_size.split(','))),
                    raw_voxel_size=tuple(
                        map(float, self.config.raw_voxel_size.split(','))),
                    src_space=self.config.src_space,
                    dest_space=self.config.orientation,
                    swap_xy=self.config.upsample_swap_xy
                )
                upsampler.run()
            except Exception as e:
                self.logger.error(f"Atlas upsampling failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 5: Upsampled atlas exists; skipping upsampling.")

    def _step_cell_labeling(self) -> None:
        if not check_file_exists(self.detected_cells_csv):
            try:
                self.logger.info(
                    "Step 6: Generating cell labels using CountGD")
                argv_backup = sys.argv.copy()
                sys.argv = [sys.argv[0]]
                from napari_ants_plugin.core.detect import run_countgd
                result = run_countgd(
                    image_path=self.zarr_signal_path,
                    shapes=None,  # Update if exemplar shapes are available
                    label_type="points",  # or "bboxes" as needed
                    text_prompt=self.config.cell_prompt,
                    confidence_threshold=0.01,
                    current_z_slice_only=False,
                    cell_size_radius=self.config.deduplication_radius,
                    csv_output_path=self.detected_cells_csv,
                )
                sys.argv = argv_backup
                detected = result.get("detected_cells", [])
                self.logger.info(
                    f"Cell labeling completed. Detected {len(detected)} cells.")
            except Exception as e:
                self.logger.error(f"Cell labeling failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 6: Cell labeling output exists; skipping labeling.")

    def _step_deduplicate_cells(self) -> None:
        if not check_file_exists(self.unique_cells_csv):
            try:
                self.logger.info("Step 7: Removing duplicate cell detections")
                df_detected = pd.read_csv(self.detected_cells_csv)
                unique_locations = remove_duplicate_cells(
                    cell_locations=df_detected[['z', 'y', 'x']].values,
                    cell_size_radius=self.config.deduplication_radius,
                )
                df_out = pd.DataFrame(
                    unique_locations, columns=['z', 'y', 'x'])
                df_out.to_csv(self.unique_cells_csv, index=False)
                self.logger.info(
                    f"Unique cell locations saved to: {self.unique_cells_csv}")
            except Exception as e:
                self.logger.error(f"Cell deduplication failed: {e}")
                sys.exit(1)
        else:
            self.logger.info(
                "Step 7: Unique cell output exists; skipping deduplication.")

    # -------------------------------------------------------------------------
    # Step 8: Launch Napari Viewer with Sparse ROI Extraction UI
    # -------------------------------------------------------------------------
    def run_viewer(self) -> None:
        """
        Set up and launch the Napari viewer with sparse ROI extraction UI.
        Uses the processed outputs:
          - Signal Zarr from self.zarr_signal_path,
          - Annotation Zarr from self.unsampled_annotation_zarr,
          - Processed points from self.unique_cells_csv,
          - Additional outputs for points_final and cell counts.
        The atlas is taken from --atlas_name.
        """
        self.logger.info(
            "Step 8: Launching Napari viewer with sparse ROI extraction UI")
        # Import UI helper functions and widgets.
        from napari_ants_plugin.regions_gpu import (
            FilteredLabels, RegionTreeWidget, create_overlay_label,
            setup_mouse_move_callback, add_points_layer, process_points,
            precompute_hierarchical_counts_recursive, save_cell_counts_by_region,
            compute_region_bounding_boxes_by_acronym
        )
        from brainglobe_atlasapi import BrainGlobeAtlas
        import zarr
        import dask.array as da
        import napari

        # Load the atlas using the same name as provided in --atlas_name.
        atlas = BrainGlobeAtlas(self.config.atlas_name, check_latest=False)

        # Load the signal image from the generated Zarr store.
        signal_z = zarr.open(self.zarr_signal_path, mode="r")
        signal_image = da.from_zarr(signal_z["data"])

        # Load the annotation data from the unsampled annotation Zarr.
        anno_z = zarr.open(self.unsampled_annotation_zarr, mode="r")
        annotation_data = da.from_zarr(anno_z["data"], chunks=(1, 1024, 1024))
        dask_anno = annotation_data

        # Process points using unique_cells_csv as the points input.
        if os.path.exists(self.unique_cells_csv) and not os.path.exists(self.points_final_csv):
            df_points = process_points(
                self.unique_cells_csv, anno_z["data"], atlas)
            if not df_points.empty:
                df_points.to_csv(self.points_final_csv, index=False)
        else:
            self.logger.info(
                f"Loading the existing points with region names file {self.points_final_csv}")
            df_points = pd.read_csv(self.points_final_csv)

        # Load or compute hierarchical cell counts.
        if os.path.exists(self.cell_counts_csv):
            self.logger.info(
                f"Loading the existing cell count file {self.cell_counts_csv}.")
            hierarchical_df = pd.read_csv(self.cell_counts_csv)
            group_counts = {row["acronym"]: row["cell_count"]
                            for row in hierarchical_df.to_dict("records")}
        else:
            save_cell_counts_by_region(
                df_points, atlas, output_file=self.cell_counts_csv)
            hierarchical_df = pd.read_csv(self.cell_counts_csv)
            group_counts = {row["acronym"]: row["cell_count"]
                            for row in hierarchical_df.to_dict("records")}

        hierarchical_counts = precompute_hierarchical_counts_recursive(
            atlas, group_counts)

        # Compute region bounding boxes from points to speed up ROI extraction.
        region_bounding_boxes = compute_region_bounding_boxes_by_acronym(
            df_points, margin=5)

        # Set up the Napari viewer.
        viewer = napari.Viewer()
        viewer.add_image(signal_image, name="Anatomical Reference",
                         colormap="gray", contrast_limits=(0, 8000))

        # Create a filtered annotation layer.
        anno_layer = FilteredLabels(
            dask_anno, name="Filtered Annotation", opacity=0.5)
        viewer.add_layer(anno_layer)

        # Create the region tree widget.
        region_tree = RegionTreeWidget(
            anno_layer=anno_layer,
            bg_tree=atlas.structures.tree,
            bg_atlas=atlas,
            hierarchical_counts=hierarchical_counts,
            dask_anno_gpu=dask_anno,
            annotation_data_cpu=dask_anno,
            viewer=viewer,
            signal_image=signal_image,
            region_bounding_boxes=region_bounding_boxes,
        )
        viewer.window.add_dock_widget(
            region_tree, name="Brain Structure Tree", area="right")

        # Add overlay label and set up mouse move callback.
        overlay_label = create_overlay_label(viewer)
        setup_mouse_move_callback(
            viewer, anno_layer, overlay_label, atlas, group_counts)
        add_points_layer(viewer, df_points)

        self.logger.info("Launching Napari event loop.")
        napari.run()


def parse_args() -> Any:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Refactored Image Processing Pipeline")
    parser.add_argument("--background", type=str, required=True,
                        help="Path to the background TIFF file.")
    parser.add_argument("--signal", type=str, required=True,
                        help="Path to the signal TIFF file.")
    parser.add_argument("--atlas_name", type=str, default="allen_mouse_25um",
                        help="BrainGlobe atlas name (used for both processing and viewer).")
    parser.add_argument("--orientation", type=str, default="ial",
                        help="Target orientation for downsampling and registration.")
    parser.add_argument("--input_orientation", type=str, default="ial",
                        help="Input orientation of the background image.")
    parser.add_argument("--output_dir", type=str, default="output_pipeline",
                        help="Directory where pipeline outputs are saved.")
    parser.add_argument("--cell_prompt", type=str, default="cell",
                        help="Text prompt for cell counting.")
    parser.add_argument("--deduplication_radius", type=float, default=5.0,
                        help="Radius for cell deduplication.")
    parser.add_argument("--log_file", type=str, default="pipeline.log",
                        help="Name of the pipeline log file.")
    parser.add_argument("--atlas_voxel_size", type=str, default="25.0,25.0,25.0",
                        help="Atlas voxel size in Z,Y,X order (comma-separated).")
    parser.add_argument("--raw_voxel_size", type=str, default="4.0,2.0,2.0",
                        help="Raw voxel size in Z,Y,X order (comma-separated).")
    parser.add_argument('--upsample_swap_xy', dest='upsample_swap_xy', action='store_true',
                        help="Swap the X and Y axes during upsampling.")
    parser.add_argument('--no_upsample_swap_xy', dest='upsample_swap_xy', action='store_false',
                        help="Do not swap the X and Y axes during upsampling.")
    parser.set_defaults(upsample_swap_xy=False)
    parser.add_argument("--src_space", type=str, default=None,
                        help="Optional source space for atlas mapping during upsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_folders = setup_output_folders(args.output_dir)
    if not os.path.isabs(args.log_file):
        args.log_file = os.path.join(output_folders["logs"], args.log_file)
    logger_inst = setup_logger(args.log_file)

    pipeline = ImageProcessingPipeline(config=args, logger=logger_inst)
    pipeline.run()         # Steps 1-7: Processing pipeline
    pipeline.run_viewer()  # Step 8: Launch Napari viewer with UI


if __name__ == "__main__":
    main()
