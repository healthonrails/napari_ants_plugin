#!/usr/bin/env python
"""
Improved Napari Brain Region Cell Count Viewer with GPU Acceleration.
This file integrates data loading, GPU/CPU computation, interactive UI elements,
and efficient real-time ROI extraction using precomputed region bounding boxes.
"""

import os
import argparse
import logging
import csv
import time
import functools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import napari
import zarr
import dask.array as da

# GPU libraries
try:
    import cupy as cp  # GPU array library
    from cupyx.scipy.ndimage import binary_erosion, label as gpu_label  # GPU-based functions
    GPU_ACCELERATED_EROSION: bool = True
    GPU_LABEL: bool = True
except ImportError:
    import cupy  # type: ignore
    from scipy.ndimage import binary_erosion, label as cpu_label  # CPU fallback
    GPU_ACCELERATED_EROSION = False
    GPU_LABEL = False

from skimage.measure import marching_cubes  # For surface extraction
from skimage.segmentation import flood  # For 2D flood fill (if needed)

from brainglobe_atlasapi import BrainGlobeAtlas

# Qt imports for UI
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox
)
from qtpy.QtGui import QFont, QCursor
from qtpy.QtCore import Qt

# Napari imports for custom layers
from napari.layers import Labels

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Profiling Decorator
# ==============================================================================


def profile_func(func):
    """
    A simple profiling decorator that logs the execution time of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# ==============================================================================
# Module: Data Processing and Utility Functions
# ==============================================================================


def compute_recursive_counts(region_id: int,
                             mapping: Dict[int, Tuple[str, str]],
                             group_counts: Dict[str, int],
                             tree: Any,
                             memo: Dict[int, int]) -> int:
    if region_id in memo:
        return memo[region_id]
    # Get the region acronym.
    acr = mapping.get(region_id, ("", ""))[0]
    # Start with the direct count for this region.
    total = group_counts.get(acr, 0)
    # Recursively add counts from all children.
    for child in tree.children(region_id):
        total += compute_recursive_counts(child.identifier,
                                          mapping, group_counts, tree, memo)
    memo[region_id] = total
    return total


def precompute_hierarchical_counts_recursive(
    atlas: BrainGlobeAtlas, group_counts: Dict[str, int]
) -> Dict[str, int]:
    mapping = create_region_mapping(atlas)
    hierarchical_counts: Dict[str, int] = {}
    tree = atlas.structures.tree
    memo: Dict[int, int] = {}
    # For each region in the atlas, compute its full (recursive) count.
    for region_id, (acr, _) in mapping.items():
        hierarchical_counts[acr] = compute_recursive_counts(
            region_id, mapping, group_counts, tree, memo)
    return hierarchical_counts


def get_bounding_box_mask(
    dask_anno_gpu: da.Array, selected_ids: List[int]
) -> Tuple[Optional[da.Array], Optional[Tuple[int, int, int, int, int, int]]]:
    """
    Compute a bounding box mask for the selected region IDs using the full annotation.

    Args:
        dask_anno_gpu: Dask array of the annotation data.
        selected_ids: List of region IDs to select.

    Returns:
        (region_mask, bounding_box) where bounding_box is (z_min, z_max, y_min, y_max, x_min, x_max).
    """
    valid_ids_gpu = cp.array(selected_ids, dtype=dask_anno_gpu.dtype)
    region_mask = da.isin(dask_anno_gpu, valid_ids_gpu)
    z_idx, y_idx, x_idx = da.where(region_mask)
    try:
        z_min, z_max, y_min, y_max, x_min, x_max = da.compute(
            z_idx.min(), z_idx.max(),
            y_idx.min(), y_idx.max(),
            x_idx.min(), x_idx.max()
        )
    except ValueError:
        return None, None
    return region_mask, (z_min, z_max, y_min, y_max, x_min, x_max)


class FilteredLabels(Labels):
    """
    A custom Napari Labels layer that filters its displayed data based on selected IDs.
    """

    def __init__(self, data: Any, **kwargs: Any):
        super().__init__(data, **kwargs)
        self._original_data = data
        self._selected_ids: Optional[List[int]] = None

    @property
    def displayed_data(self) -> Any:
        if self._selected_ids is not None:
            mask = np.isin(self._original_data, self._selected_ids)
            return np.where(mask, self._original_data, 0)
        return self._original_data

    @displayed_data.setter
    def displayed_data(self, value: Any) -> None:
        self._original_data = value
        self._selected_ids = None
        self._data = value
        self.refresh()

    def set_selected_ids(self, selected_ids: List[int]) -> None:
        self._selected_ids = selected_ids
        self.refresh()

    def reset_filter(self) -> None:
        self._selected_ids = None
        self.refresh()

# ------------------------------------------------------------------------------
# New ROI Extraction Function (Fallback Method)
# ------------------------------------------------------------------------------


@profile_func
def extract_region_roi(
    signal_image: da.Array,
    dask_anno_gpu: da.Array,
    selected_region: int
) -> Optional[np.ndarray]:
    """
    Extract the ROI corresponding to the selected region from the anatomical reference.
    (Fallback method that uses the full annotation data.)

    Args:
        signal_image: The anatomical reference as a Dask array.
        dask_anno_gpu: The annotation data as a Dask array (GPU-backed).
        selected_region: The label of the region to extract.

    Returns:
        A NumPy array of the extracted region (with values outside the region set to 0),
        or None if the region is not found.
    """
    mask, bbox = get_bounding_box_mask(dask_anno_gpu, [selected_region])
    if mask is None or bbox is None:
        return None

    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    logger.info(
        f"Fallback region bounding box for region {selected_region}: {bbox}")

    try:
        roi_signal = signal_image[z_min:z_max, y_min:y_max, x_min:x_max]
        roi_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        roi_signal = roi_signal.persist()
        roi_mask = roi_mask.persist()
        roi_signal_np = roi_signal.compute()
        roi_mask_np = roi_mask.compute()
        if hasattr(roi_mask_np, "get"):
            roi_mask_np = roi_mask_np.get()
    except Exception as err:
        logger.error("Error computing fallback ROI: %s", err)
        return None

    extracted_region = np.where(roi_mask_np, roi_signal_np, 0)
    return extracted_region

# ------------------------------------------------------------------------------
# Precomputed Bounding Boxes from Points
# ------------------------------------------------------------------------------


def compute_region_bounding_boxes_by_acronym(
    df: pd.DataFrame, margin: int = 5
) -> Dict[str, Tuple[int, int, int, int, int, int]]:
    """
    Compute bounding boxes (with an optional margin) for each region based on point coordinates.
    The bounding boxes are keyed by region acronym.

    Args:
        df: DataFrame with columns "z", "y", "x", and "region_acronym".
        margin: Additional voxels to add around the computed bounding box.

    Returns:
        Dictionary mapping region_acronym to (z_min, z_max, y_min, y_max, x_min, x_max).
    """
    boxes = {}
    for acr, group in df.groupby("region_acronym"):
        if pd.isna(acr):
            continue
        z_min = int(group["z"].min()) - margin
        z_max = int(group["z"].max()) + margin
        y_min = int(group["y"].min()) - margin
        y_max = int(group["y"].max()) + margin
        x_min = int(group["x"].min()) - margin
        x_max = int(group["x"].max()) + margin
        boxes[acr] = (max(z_min, 0), z_max, max(
            y_min, 0), y_max, max(x_min, 0), x_max)
    return boxes

# ------------------------------------------------------------------------------
# Other Data Processing Functions
# ------------------------------------------------------------------------------


def load_atlas_and_data(
    atlas_name: str,
    signal_zarr_path: str,
    anno_zarr_path: str
) -> Tuple[BrainGlobeAtlas, Any, Any, Any]:
    """
    Load the BrainGlobeAtlas and the signal/annotation data from Zarr files.
    """
    logger.info(f"Loading BrainGlobeAtlas: {atlas_name}")
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    logger.info(f"Loading signal image from: {signal_zarr_path}")
    signal_z = zarr.open(signal_zarr_path, mode="r")
    signal_image = signal_z["data"]
    logger.info(f"Loading annotation data from: {anno_zarr_path}")
    anno_z = zarr.open(anno_zarr_path, mode="r")
    annotation_data = anno_z["data"]
    return atlas, signal_image, annotation_data, annotation_data


def create_region_mapping(bg_atlas: BrainGlobeAtlas) -> Dict[int, Tuple[str, str]]:
    """
    Create a mapping from region ID to (acronym, name).
    """
    lookup_df = bg_atlas.lookup_df
    return {int(row["id"]): (row["acronym"], row["name"]) for _, row in lookup_df.iterrows()}


def add_region_info_to_points(df: pd.DataFrame, annotation: np.ndarray, bg_atlas: BrainGlobeAtlas) -> pd.DataFrame:
    x = df["x"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    z = df["z"].to_numpy(dtype=int)
    shape = annotation.shape  # (Z, Y, X)

    valid_mask = (z >= 0) & (z < shape[0]) & (y >= 0) & (
        y < shape[1]) & (x >= 0) & (x < shape[2])
    if not np.all(valid_mask):
        logger.info("Filtered out %d out-of-bound points.",
                    np.sum(~valid_mask))
    df = df.loc[valid_mask].copy()
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    region_vals = annotation[z, y, x]
    mapping = create_region_mapping(bg_atlas)
    region_info = pd.Series(region_vals).map(mapping)
    df["region_acronym"] = region_info.apply(
        lambda x: x[0] if isinstance(x, tuple) else None)
    df["region_name"] = region_info.apply(
        lambda x: x[1] if isinstance(x, tuple) else None)
    return df


def process_points(
    csv_path: str,
    annotation: np.ndarray,
    bg_atlas: BrainGlobeAtlas
) -> pd.DataFrame:
    """
    Process the points CSV and add region info.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"{csv_path} not found. Skipping point processing.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded Points CSV from {csv_path}")
    logger.info(f"Points CSV head:\n{df.head()}")
    df = add_region_info_to_points(df, annotation, bg_atlas)
    df_filtered = df[df["region_acronym"].notna() & (
        df["region_acronym"] != "Unknown")].copy()
    return df_filtered


def precompute_hierarchical_counts(
    atlas: BrainGlobeAtlas, group_counts: Dict[str, int]
) -> Dict[str, int]:
    """
    Precompute hierarchical cell counts for each region.
    """
    mapping = create_region_mapping(atlas)
    acronym_to_id = {v[0]: k for k, v in mapping.items()}
    descendant_cache: Dict[int, List[int]] = {}

    def cached_get_descendants(region_id: int) -> List[int]:
        if region_id in descendant_cache:
            return descendant_cache[region_id]
        descendants = atlas.get_structure_descendants(region_id)
        descendant_cache[region_id] = descendants
        return descendants

    hierarchical_counts: Dict[str, int] = {}
    id_to_acronym = {k: v[0] for k, v in mapping.items()}
    for acr, region_id in acronym_to_id.items():
        descendants = cached_get_descendants(region_id)
        descendant_acrs = [id_to_acronym.get(
            did) for did in descendants if did in id_to_acronym]
        hierarchical_counts[acr] = group_counts.get(
            acr, 0) + sum(group_counts.get(d, 0) for d in descendant_acrs)
    return hierarchical_counts


def count_points_in_region_hierarchical(
    group_counts: Dict[str, int],
    region_acronym: Optional[str],
    atlas: BrainGlobeAtlas
) -> int:
    """
    Count points in a region including its descendants.
    """
    if region_acronym is None:
        return 0
    mapping = create_region_mapping(atlas)
    acronym_to_id = {v[0]: k for k, v in mapping.items()}
    try:
        region_id = acronym_to_id[region_acronym]
    except KeyError:
        logger.warning(
            f"Region acronym {region_acronym} not found in mapping.")
        return group_counts.get(region_acronym, 0)
    descendant_ids = atlas.get_structure_descendants(region_id)
    id_to_acronym = {k: v[0] for k, v in mapping.items()}
    descendant_acrs = [id_to_acronym.get(
        did) for did in descendant_ids if did in id_to_acronym]
    total = group_counts.get(region_acronym, 0)
    for acr in descendant_acrs:
        total += group_counts.get(acr, 0)
    return total


def save_cell_counts_by_region(
    df: pd.DataFrame,
    atlas: BrainGlobeAtlas,
    output_file: str = "cell_counts_by_region.csv"
) -> None:
    """
    Save hierarchical cell counts to a CSV file.
    """
    raw_group_counts = df.groupby("region_acronym").size().to_dict()
    all_counts = []
    for _, row in atlas.lookup_df.iterrows():
        acr = row["acronym"]
        count = count_points_in_region_hierarchical(
            raw_group_counts, acr, atlas)
        all_counts.append(
            {"acronym": acr, "name": row["name"], "cell_count": count})
    counts_df = pd.DataFrame(all_counts)
    counts_df.to_csv(output_file, index=False)
    logger.info(f"Saved cell counts for each region to '{output_file}'.")

# ==============================================================================
# Module: UI Helper Functions and Widgets
# ==============================================================================


def get_structure_info(
    structure_id: Union[int, str],
    atlas: BrainGlobeAtlas
) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve the (acronym, name) tuple for a given structure ID.
    """
    try:
        structure_id_int = int(structure_id)
    except (ValueError, TypeError):
        return (None, None)
    mapping = create_region_mapping(atlas)
    return mapping.get(structure_id_int, (None, None))


def add_points_layer(viewer: napari.Viewer, df: pd.DataFrame) -> None:
    """
    Add a points layer to the napari viewer.
    """
    if df.empty:
        return
    pts = df[['z', 'y', 'x']].to_numpy()
    viewer.add_points(pts, name="Points", size=10,
                      face_color="red", border_color="white")


def in_layer(layer: Any, world_coord: Tuple[float, float]) -> bool:
    """
    Check whether a world coordinate is within a layer.
    """
    data_coord = layer.world_to_data(world_coord)
    data_idx = np.round(data_coord).astype(int)
    if np.any(data_idx < 0) or np.any(data_idx >= np.array(layer.data.shape)):
        return False
    return True


def create_overlay_label(viewer: napari.Viewer) -> QLabel:
    """
    Create an overlay QLabel for displaying information on hover.
    """
    label = QLabel(viewer.window.qt_viewer.canvas.native)
    label.setStyleSheet(
        """
        background-color: rgba(0, 0, 0, 150);
        color: white;
        padding: 3px;
        border: 1px solid rgba(255, 255, 255, 180);
        border-radius: 3px;
        """
    )
    label.setFont(QFont("SansSerif", 10))
    label.setAttribute(Qt.WA_TransparentForMouseEvents)
    label.show()
    return label


def setup_mouse_move_callback(
    viewer: napari.Viewer,
    anno_layer: Any,
    label: QLabel,
    atlas: BrainGlobeAtlas,
    group_counts: Dict[str, int]
) -> None:
    """
    Setup a callback to update the overlay label when the mouse moves.
    """
    @viewer.mouse_move_callbacks.append
    def update_cursor_info(viewer: napari.Viewer, event: Any) -> None:
        if in_layer(anno_layer, event.position):
            structure_id = anno_layer.get_value(event.position)
            acronym, structure_name = get_structure_info(structure_id, atlas)
            if acronym is not None:
                count = count_points_in_region_hierarchical(
                    group_counts, acronym, atlas)
                text = (f"Structure: {acronym}\n"
                        f"Name: {structure_name}\n"
                        f"ID: {structure_id}\n"
                        f"Cell count: {count}")
            else:
                text = ""
            label.setText(text)
            label.adjustSize()
            global_pos = QCursor.pos()
            local_pos = viewer.window.qt_viewer.canvas.native.mapFromGlobal(
                global_pos)
            label.move(local_pos.x() + 10, local_pos.y() + 10)
            label.show()
        else:
            label.hide()


class RegionGrowingDialog(QDialog):
    """
    A dialog to adjust region growing parameters.
    """

    def __init__(self, seed_coords: Tuple[int, int, int], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Region Growing Parameters")
        self.seed_coords = seed_coords

        seed_label = QLabel(f"Seed Coordinates: {seed_coords}")

        margin_label = QLabel("ROI Margin:")
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 200)
        self.margin_spin.setValue(50)

        tolerance_label = QLabel("Tolerance:")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0, 10)
        self.tolerance_spin.setValue(0)

        param_layout = QVBoxLayout()
        param_layout.addWidget(seed_label)
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(margin_label)
        h_layout1.addWidget(self.margin_spin)
        param_layout.addLayout(h_layout1)
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(tolerance_label)
        h_layout2.addWidget(self.tolerance_spin)
        param_layout.addLayout(h_layout2)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        param_layout.addWidget(self.button_box)
        self.setLayout(param_layout)

    def getParameters(self) -> Dict[str, Union[int, float]]:
        return {
            "margin": self.margin_spin.value(),
            "tolerance": self.tolerance_spin.value(),
        }


class RegionTreeWidget(QWidget):
    """
    A tree widget displaying brain regions that triggers ROI extraction on double-click.
    Now also displays cell/point markers within the extracted ROI.

    It also displays cell counts and, if available,
    cell density information. Works both with hierarchical_counts as just counts (int)
    or as a tuple (cell_count, cell_density).
    """

    def __init__(
        self,
        anno_layer: Any,
        bg_tree: Any,
        bg_atlas: BrainGlobeAtlas,
        hierarchical_counts: Dict[str, Union[int, Tuple[int, float]]],
        dask_anno_gpu: da.Array,
        annotation_data_cpu: da.Array,
        viewer: napari.Viewer,
        signal_image: da.Array,
        region_bounding_boxes: Dict[str, Tuple[int, int, int, int, int, int]],
        df_points: pd.DataFrame
    ) -> None:
        super().__init__()
        self.anno_layer = anno_layer
        self.bg_tree = bg_tree
        self.bg_atlas = bg_atlas
        self.hierarchical_counts = hierarchical_counts
        self.dask_anno_gpu = dask_anno_gpu
        self.annotation_data_cpu = annotation_data_cpu.rechunk("auto")
        self.viewer = viewer
        self.signal_image = signal_image
        self.region_bounding_boxes = region_bounding_boxes
        self.mapping = create_region_mapping(self.bg_atlas)
        self.df_points = df_points

        # Determine whether cell density is available.
        self.include_density = False
        if hierarchical_counts:
            sample_val = next(iter(hierarchical_counts.values()))
            if isinstance(sample_val, tuple) and len(sample_val) >= 2:
                self.include_density = True

        self.initUI()

    def initUI(self) -> None:
        layout = QVBoxLayout(self)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search region or name...")
        self.search_bar.textChanged.connect(self.filter_tree)
        layout.addWidget(self.search_bar)

        self.tree_widget = QTreeWidget()
        # Use a different header depending on whether density is provided.
        if self.include_density:
            self.tree_widget.setHeaderLabels(
                ["Region", "Name", "Cell Count", "Cell Density"])
        else:
            self.tree_widget.setHeaderLabels(["Region", "Name", "Cell Count"])

        self.tree_widget.setSortingEnabled(True)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setStyleSheet(
            """
            QTreeWidget {
                background-color: #f2f2f2;
                alternate-background-color: #e6e6e6;
                font: 12px "Segoe UI";
                border: 1px solid #cccccc;
                color: #333333;
            }
            QTreeWidget::item { padding: 4px; }
            QTreeWidget::item:selected { background-color: #0078d7; color: #ffffff; }
            QHeaderView::section {
                background-color: #0078d7;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #cccccc;
                font-weight: bold;
            }
            """
        )
        layout.addWidget(self.tree_widget)
        self.populate_tree()
        self.tree_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tree_widget.itemClicked.connect(self.on_item_clicked)

        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Show All Regions")
        self.reset_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                font: 12px "Segoe UI";
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #005fa3; }
            """
        )
        self.reset_button.clicked.connect(self.reset_filter)
        button_layout.addWidget(self.reset_button)

        self.export_button = QPushButton("Export Table")
        self.export_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                font: 12px "Segoe UI";
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #005fa3; }
            """
        )
        self.export_button.clicked.connect(self.export_table)
        button_layout.addWidget(self.export_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_region_display(self, region_acronym: str, region_id: str) -> str:
        return f"{region_acronym}({region_id})"

    def get_region_name(self, region_id: str) -> str:
        try:
            reg_id_int = int(region_id)
            return self.mapping.get(reg_id_int, ("", ""))[1]
        except (ValueError, TypeError):
            return ""

    def get_region_count(self, region_acronym: str) -> int:
        value = self.hierarchical_counts.get(region_acronym)
        if isinstance(value, tuple):
            return value[0]
        return value if value is not None else 0

    def get_region_density(self, region_acronym: str) -> float:
        value = self.hierarchical_counts.get(region_acronym)
        if isinstance(value, tuple) and len(value) >= 2:
            return value[1]
        return 0.0

    def populate_tree(self) -> None:
        self.tree_widget.clear()
        root_id = self.bg_tree.root
        root_node = self.bg_tree.get_node(root_id)
        pure_acronym = str(root_node.tag).split(" (")[0]
        region_id = ""
        if root_node.data:
            region_id = root_node.data.get("id", "")
        if not region_id:
            region_id = str(root_node.identifier)
        region_display = self.get_region_display(pure_acronym, region_id)
        region_name = self.get_region_name(region_id)
        region_count = self.get_region_count(pure_acronym)
        # Depending on the flag, populate with or without density.
        if self.include_density:
            region_density = self.get_region_density(pure_acronym)
            root_item = QTreeWidgetItem(
                [region_display, region_name, str(
                    region_count), f"{region_density:.2f}"]
            )
        else:
            root_item = QTreeWidgetItem(
                [region_display, region_name, str(region_count)]
            )
        root_item.setData(0, Qt.UserRole, region_id)
        root_item.setData(0, Qt.UserRole + 1, pure_acronym)
        self.tree_widget.addTopLevelItem(root_item)
        self.add_children(root_item, root_node)
        self.tree_widget.expandAll()

    def add_children(self, parent_item: QTreeWidgetItem, parent_node: Any) -> None:
        for child in self.bg_tree.children(parent_node.identifier):
            pure_acronym = str(child.tag).split(" (")[0]
            region_id = ""
            if child.data:
                region_id = child.data.get("id", "")
            if not region_id:
                region_id = str(child.identifier)
            region_display = self.get_region_display(pure_acronym, region_id)
            region_name = self.get_region_name(region_id)
            region_count = self.get_region_count(pure_acronym)
            if self.include_density:
                region_density = self.get_region_density(pure_acronym)
                child_item = QTreeWidgetItem(
                    [region_display, region_name, str(
                        region_count), f"{region_density:.2f}"]
                )
            else:
                child_item = QTreeWidgetItem(
                    [region_display, region_name, str(region_count)]
                )
            child_item.setData(0, Qt.UserRole, region_id)
            child_item.setData(0, Qt.UserRole + 1, pure_acronym)
            parent_item.addChild(child_item)
            self.add_children(child_item, child)

    def filter_tree(self, text: str) -> None:
        text = text.lower().strip()

        def filter_item(item: QTreeWidgetItem) -> bool:
            match = text in item.text(
                0).lower() or text in item.text(1).lower()
            child_match = any(filter_item(item.child(i))
                              for i in range(item.childCount()))
            item.setHidden(not (match or child_match))
            return match or child_match

        for i in range(self.tree_widget.topLevelItemCount()):
            filter_item(self.tree_widget.topLevelItem(i))

    def export_table(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Table to CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        rows = [["Region", "Name", "Cell Count"]]

        def traverse(item: QTreeWidgetItem) -> None:
            if not item.isHidden():
                rows.append([item.text(0), item.text(1), item.text(2)])
                for i in range(item.childCount()):
                    traverse(item.child(i))

        for i in range(self.tree_widget.topLevelItemCount()):
            traverse(self.tree_widget.topLevelItem(i))
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        logger.info(f"Exported table to {file_path}")

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        When a region is double-clicked, extract that region from the anatomical reference.
        Then, filter and display the corresponding cell/point markers within the ROI.
        """
        try:
            selected_region = int(item.data(0, Qt.UserRole))
        except (ValueError, TypeError) as err:
            logger.error("Invalid region id: %s", err)
            QMessageBox.critical(self, "Error", "Invalid region selected.")
            return

        logger.info(f"Selected region id: {selected_region}")
        # Get region acronym from mapping.
        acr, _ = self.mapping.get(selected_region, (None, None))

        # Determine the bounding box and extract ROI.
        if acr is not None and acr in self.region_bounding_boxes:
            bbox = self.region_bounding_boxes[acr]
            logger.info(
                f"Using precomputed bounding box for region {acr}: {bbox}")
            z_min, z_max, y_min, y_max, x_min, x_max = bbox
            try:
                roi = self.signal_image[z_min:z_max, y_min:y_max, x_min:x_max]
                roi_mask = self.dask_anno_gpu[z_min:z_max,
                                              y_min:y_max, x_min:x_max]
                roi_mask_np = roi_mask.compute() == selected_region
                if hasattr(roi, "compute"):
                    extracted_region = roi.compute()
                else:
                    extracted_region = roi
                extracted_region = np.where(roi_mask_np, extracted_region, 0)
            except Exception as err:
                logger.error(
                    "Error computing ROI from precomputed bounding box: %s", err)
                QMessageBox.critical(
                    self, "Error", f"Error computing ROI: {err}")
                return
            region_label = f"Extracted Region {acr}"
        else:
            logger.info(
                "Falling back to full annotation-based ROI extraction.")
            extracted_region = extract_region_roi(
                self.signal_image, self.dask_anno_gpu, selected_region)
            region_label = f"Extracted Region {selected_region}"
            if extracted_region is None:
                logger.warning("Region not found or extraction failed.")
                QMessageBox.warning(
                    self, "Warning", "Region not found in annotation data.")
                return
            # For fallback, re-obtain the bounding box for filtering points.
            _, bbox = get_bounding_box_mask(
                self.dask_anno_gpu, [selected_region])
            if bbox is None:
                QMessageBox.warning(
                    self, "Warning", "Could not determine bounding box for ROI points.")
                return
            z_min, z_max, y_min, y_max, x_min, x_max = bbox

        # Add the extracted ROI image to the viewer.
        self.viewer.add_image(
            extracted_region,
            name=region_label,
            colormap="gray",
            contrast_limits=(max(0, np.min(extracted_region)),
                             min(8000, np.max(extracted_region))),
            translate=(z_min, y_min, x_min)
        )

        # ---- Filter and display points within the ROI ----
        if not self.df_points.empty:
            roi_points = self.df_points[
                (self.df_points["z"] >= z_min) & (self.df_points["z"] < z_max) &
                (self.df_points["y"] >= y_min) & (self.df_points["y"] < y_max) &
                (self.df_points["x"] >= x_min) & (self.df_points["x"] < x_max)
            ]
            if not roi_points.empty:
                # Adjust coordinates relative to the ROI origin.
                roi_points_coords = roi_points[[
                    'z', 'y', 'x']].to_numpy() - [z_min, y_min, x_min]
                # Convert coordinates to integer indices for indexing the ROI mask.
                indices = roi_points_coords.astype(int)
                # Use the ROI mask to select only points inside the region.
                inside_mask = roi_mask_np[indices[:, 0],
                                          indices[:, 1], indices[:, 2]]
                points_inside_roi = roi_points_coords[inside_mask]
                # Add these points as a separate layer.
                self.viewer.add_points(
                    points_inside_roi,
                    name="ROI Cells",
                    size=5,
                    face_color='red',
                    border_color="white",
                    translate=(z_min, y_min, x_min),
                )
        # ---------------------------------------------------------------

    def on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        try:
            selected_id = int(item.data(0, Qt.UserRole))
        except (ValueError, TypeError):
            return
        self.anno_layer.selected_label = selected_id if selected_id else 1
        self.anno_layer.show_selected_label = True

    def reset_filter(self) -> None:
        self.anno_layer.reset_filter()
        self.search_bar.clear()

        def reset_item(item: QTreeWidgetItem) -> None:
            item.setHidden(False)
            for i in range(item.childCount()):
                reset_item(item.child(i))
        for i in range(self.tree_widget.topLevelItemCount()):
            reset_item(self.tree_widget.topLevelItem(i))

# ==============================================================================
# Module: Argument Parsing and Main Function
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Napari viewer for brain region cell counts (GPU w/ single-pass bounding box)."
    )
    parser.add_argument(
        "--atlas", type=str, default="kim_mouse_25um", help="Name of the atlas to load")
    parser.add_argument("--signal_zarr", type=str,
                        default="MF1wt_125F_W_BS_640.zarr", help="Path to the signal zarr file")
    parser.add_argument("--anno_zarr", type=str, default="resampled_full_anno.zarr",
                        help="Path to the annotation zarr file")
    parser.add_argument("--points_csv", type=str, default="points.csv",
                        help="Path to the CSV file with points")
    parser.add_argument("--points_final_csv", type=str, default="points_final.csv",
                        help="Path to the processed points CSV file")
    parser.add_argument("--cell_counts_csv", type=str,
                        default="cell_counts_by_region.csv", help="Output CSV file for cell counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Starting GPU-based app. Args: {args}")

    try:
        atlas, signal_image, annotation_data_cpu, _ = load_atlas_and_data(
            args.atlas, args.signal_zarr, args.anno_zarr
        )
    except Exception as err:
        logger.error(f"Failed to load atlas or data: {err}")
        return

    # Wrap annotation with Dask and set chunk sizes.
    dask_anno_cpu = da.from_zarr(annotation_data_cpu, chunks=(1, 1024, 1024))

    # Transfer annotation data to GPU (lazily).
    logger.info("Transferring annotation data to GPU (lazily) ...")
    dask_anno_gpu = dask_anno_cpu.map_blocks(cp.asarray)

    # Process points.
    if os.path.exists(args.points_final_csv):
        logger.info(
            f"Processed points CSV '{args.points_final_csv}' exists. Loading.")
        df_points = pd.read_csv(args.points_final_csv)
    else:
        df_points = process_points(args.points_csv, annotation_data_cpu, atlas)
        if not df_points.empty:
            df_points.to_csv(args.points_final_csv, index=False)
            logger.info(
                f"Saved processed points as '{args.points_final_csv}'.")
        else:
            logger.warning("No valid points data loaded.")
            df_points = pd.DataFrame()

    # Load or compute cell counts.
    if os.path.exists(args.cell_counts_csv):
        logger.info(
            f"Cell counts CSV '{args.cell_counts_csv}' exists. Loading.")
        hierarchical_df = pd.read_csv(args.cell_counts_csv)
        group_counts = {row["acronym"]: row["cell_count"]
                        for row in hierarchical_df.to_dict("records")}
    else:
        if not df_points.empty:
            save_cell_counts_by_region(
                df_points, atlas, output_file=args.cell_counts_csv)
            hierarchical_df = pd.read_csv(args.cell_counts_csv)
            group_counts = {row["acronym"]: row["cell_count"]
                            for row in hierarchical_df.to_dict("records")}
        else:
            group_counts = {}

    # hierarchical_counts = precompute_hierarchical_counts(atlas, group_counts)
    hierarchical_counts = precompute_hierarchical_counts_recursive(
        atlas, group_counts)

    # Compute region bounding boxes from points to speed up ROI extraction.
    region_bounding_boxes = compute_region_bounding_boxes_by_acronym(
        df_points, margin=5)

    # Create napari viewer.
    viewer = napari.Viewer()
    viewer.add_image(signal_image, name="Anatomical Reference",
                     colormap="gray", contrast_limits=(0, 8000))
    anno_layer = FilteredLabels(
        dask_anno_cpu, name="Filtered Annotation", opacity=0.5)
    viewer.add_layer(anno_layer)

    # Add the region tree widget (pass the precomputed bounding boxes and cell points DataFrame).
    region_tree = RegionTreeWidget(
        anno_layer=anno_layer,
        bg_tree=atlas.structures.tree,
        bg_atlas=atlas,
        hierarchical_counts=hierarchical_counts,
        dask_anno_gpu=dask_anno_gpu,
        annotation_data_cpu=dask_anno_cpu,
        viewer=viewer,
        signal_image=signal_image,
        region_bounding_boxes=region_bounding_boxes,
        df_points=df_points  # Pass the points data.
    )
    viewer.window.add_dock_widget(
        region_tree, name="Brain Structure Tree (GPU, Single-Pass BBox)", area="right")

    # On-hover overlay.
    overlay_label = create_overlay_label(viewer)
    setup_mouse_move_callback(
        viewer, anno_layer, overlay_label, atlas, group_counts)

    # Add any points.
    add_points_layer(viewer, df_points)

    logger.info("Starting Napari event loop with real-time ROI extraction.")
    napari.run()


if __name__ == "__main__":
    main()
