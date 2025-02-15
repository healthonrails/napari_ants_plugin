import os
import argparse
import logging
import csv
import numpy as np
import pandas as pd
import napari
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas

# Qt imports
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QLineEdit, QFileDialog
)
from qtpy.QtGui import QFont, QCursor
from qtpy.QtCore import Qt

# Napari imports
from napari.layers import Labels

# ------------------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Custom Labels Layer
# ------------------------------------------------------------------------------


class FilteredLabels(Labels):
    """
    Custom Labels layer that shows a filtered version of the annotation data.
    When a set of region IDs is specified, only those regions are visible.
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._original_data = data
        self._selected_ids = None

    @property
    def displayed_data(self):
        if self._selected_ids is not None:
            mask = np.isin(self._original_data, self._selected_ids)
            return np.where(mask, self._original_data, 0)
        return self._original_data

    @displayed_data.setter
    def displayed_data(self, value):
        self._original_data = value
        self._selected_ids = None
        self._data = value
        self.refresh()

    def set_selected_ids(self, selected_ids):
        self._selected_ids = selected_ids
        self.refresh()

    def reset_filter(self):
        self._selected_ids = None
        self.refresh()

# ------------------------------------------------------------------------------
# Data Loading & Processing Functions
# ------------------------------------------------------------------------------


def load_atlas_and_data(atlas_name: str, signal_zarr_path: str, anno_zarr_path: str):
    logger.info("Loading BrainGlobeAtlas %s", atlas_name)
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)

    logger.info("Loading anatomical image from %s", signal_zarr_path)
    signal_zarr = zarr.open(signal_zarr_path, mode="r")
    signal_image = signal_zarr["data"]

    logger.info("Loading annotation data from %s", anno_zarr_path)
    anno_zarr = zarr.open(anno_zarr_path, mode="r")
    annotation_data = anno_zarr["data"]
    original_annotation = annotation_data

    return atlas, signal_image, annotation_data, original_annotation


def create_region_mapping(bg_atlas: BrainGlobeAtlas) -> dict:
    """Return a dictionary mapping region id (int) to (acronym, name)."""
    return {
        int(row["id"]): (row["acronym"], row["name"])
        for _, row in bg_atlas.lookup_df.iterrows()
    }


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


def process_points(csv_path: str, annotation: np.ndarray, bg_atlas: BrainGlobeAtlas) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logger.warning("%s not found. Skipping point processing.", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    logger.info("Loaded Points CSV from %s", csv_path)
    logger.info("Points CSV head:\n%s", df.head())
    df = add_region_info_to_points(df, annotation, bg_atlas)
    df_filtered = df[df["region_acronym"].notna() & (
        df["region_acronym"] != "Unknown")].copy()
    return df_filtered

# ------------------------------------------------------------------------------
# Precompute Hierarchical Counts with Caching
# ------------------------------------------------------------------------------


def precompute_hierarchical_counts(atlas: BrainGlobeAtlas, group_counts: dict) -> dict:
    """
    Precompute a dictionary mapping region acronyms to their hierarchical cell counts.
    Caches descendant lookups for speed.
    """
    mapping = create_region_mapping(atlas)  # id -> (acronym, name)
    acronym_to_id = {v[0]: k for k, v in mapping.items()}
    descendant_cache = {}

    def cached_get_descendants(region_id):
        if region_id in descendant_cache:
            return descendant_cache[region_id]
        else:
            descendants = atlas.get_structure_descendants(region_id)
            descendant_cache[region_id] = descendants
            return descendants

    hierarchical_counts = {}
    for acronym, region_id in acronym_to_id.items():
        descendants = cached_get_descendants(region_id)
        id_to_acronym = {k: v[0] for k, v in mapping.items()}
        descendant_acrs = [id_to_acronym.get(
            did) for did in descendants if id_to_acronym.get(did)]
        hierarchical_counts[acronym] = group_counts.get(
            acronym, 0) + sum(group_counts.get(d, 0) for d in descendant_acrs)
    return hierarchical_counts

# ------------------------------------------------------------------------------
# GroupBy-based Hierarchical Counting Function (fallback)
# ------------------------------------------------------------------------------


def count_points_in_region_hierarchical(group_counts: dict, region_acronym: str, atlas: BrainGlobeAtlas) -> int:
    if region_acronym is None:
        return 0
    mapping = create_region_mapping(atlas)
    acronym_to_id = {v[0]: k for k, v in mapping.items()}
    try:
        region_id = acronym_to_id[region_acronym]
    except KeyError:
        logger.warning(
            "Region acronym %s not found in mapping.", region_acronym)
        return group_counts.get(region_acronym, 0)
    descendant_ids = atlas.get_structure_descendants(region_id)
    id_to_acronym = {k: v[0] for k, v in mapping.items()}
    descendant_acrs = [id_to_acronym.get(
        did) for did in descendant_ids if id_to_acronym.get(did)]
    total = group_counts.get(region_acronym, 0)
    for acr in descendant_acrs:
        total += group_counts.get(acr, 0)
    return total


def save_cell_counts_by_region(df: pd.DataFrame, atlas: BrainGlobeAtlas, output_file: str = "cell_counts_by_region.csv"):
    raw_group_counts = df.groupby("region_acronym").size().to_dict()
    all_counts = []
    for _, row in atlas.lookup_df.iterrows():
        acr = row["acronym"]
        count = count_points_in_region_hierarchical(
            raw_group_counts, acr, atlas)
        all_counts.append({
            "acronym": acr,
            "name": row["name"],
            "cell_count": count
        })
    counts_df = pd.DataFrame(all_counts)
    counts_df.to_csv(output_file, index=False)
    logger.info("Saved cell counts for each region to '%s'.", output_file)

# ------------------------------------------------------------------------------
# UI Helper Functions
# ------------------------------------------------------------------------------


def get_structure_info(structure_id, atlas: BrainGlobeAtlas):
    try:
        structure_id = int(structure_id)
    except (ValueError, TypeError):
        return (None, None)
    mapping = create_region_mapping(atlas)
    return mapping.get(structure_id, (None, None))


def add_points_layer(viewer, df: pd.DataFrame):
    if df.empty:
        return
    points = df[['z', 'y', 'x']].to_numpy()
    viewer.add_points(points, name="Points", size=10,
                      face_color="red", border_color="white")


def in_layer(layer, world_coord):
    data_coord = layer.world_to_data(world_coord)
    data_idx = np.round(data_coord).astype(int)
    if np.any(data_idx < 0) or np.any(data_idx >= np.array(layer.data.shape)):
        return False
    return True


def create_overlay_label(viewer) -> QLabel:
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


def setup_mouse_move_callback(viewer, anno_layer, label, atlas, group_counts):
    @viewer.mouse_move_callbacks.append
    def update_cursor_info(viewer, event):
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

# ------------------------------------------------------------------------------
# Region Tree Widget for Filtering the Annotation Layer with Search and Three Columns
# ------------------------------------------------------------------------------


class RegionTreeWidget(QWidget):
    def __init__(self, anno_layer, bg_tree, bg_atlas, hierarchical_counts: dict):
        super().__init__()
        self.anno_layer = anno_layer
        self.bg_tree = bg_tree
        self.bg_atlas = bg_atlas
        # Precomputed hierarchical counts dictionary
        self.hierarchical_counts = hierarchical_counts
        self.mapping = create_region_mapping(
            self.bg_atlas)  # id -> (acronym, name)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Add search bar for filtering
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search region or name...")
        self.search_bar.textChanged.connect(self.filter_tree)
        layout.addWidget(self.search_bar)

        self.tree_widget = QTreeWidget()
        # Three columns: Region (Acronym(ID)), Name, Cell Count
        self.tree_widget.setHeaderLabels(["Region", "Name", "Cell Count"])
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
        self.tree_widget.setAlternatingRowColors(True)
        layout.addWidget(self.tree_widget)
        self.populate_tree()
        self.tree_widget.itemClicked.connect(self.on_item_clicked)

        # Create a horizontal layout for buttons
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
        """Return display string for column 0: Acronym(ID)"""
        return f"{region_acronym}({region_id})"

    def get_region_name(self, region_id: str) -> str:
        """Return region name based on region_id using the mapping; empty string if not found."""
        try:
            reg_id_int = int(region_id)
            return self.mapping.get(reg_id_int, ("", ""))[1]
        except:
            return ""

    def get_region_count(self, region_acronym: str) -> int:
        return self.hierarchical_counts.get(region_acronym, 0)

    def populate_tree(self):
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
        root_item = QTreeWidgetItem(
            [region_display, region_name, str(region_count)])
        root_item.setData(0, Qt.UserRole, region_id)
        root_item.setData(0, Qt.UserRole + 1, pure_acronym)
        self.tree_widget.addTopLevelItem(root_item)
        self.add_children(root_item, root_node)
        self.tree_widget.expandAll()

    def add_children(self, parent_item, parent_node):
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
            child_item = QTreeWidgetItem(
                [region_display, region_name, str(region_count)])
            child_item.setData(0, Qt.UserRole, region_id)
            child_item.setData(0, Qt.UserRole + 1, pure_acronym)
            parent_item.addChild(child_item)
            self.add_children(child_item, child)

    def filter_tree(self, text):
        text = text.lower().strip()

        def filter_item(item):
            # Check if search text is in the Region (column 0) or Name (column 1)
            match = text in item.text(
                0).lower() or text in item.text(1).lower()
            child_match = False
            for i in range(item.childCount()):
                child = item.child(i)
                if filter_item(child):
                    child_match = True
            item.setHidden(not (match or child_match))
            return match or child_match
        for i in range(self.tree_widget.topLevelItemCount()):
            filter_item(self.tree_widget.topLevelItem(i))

    def export_table(self):
        # Prompt user to choose file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Table to CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        rows = []
        # Write header row
        rows.append(["Region", "Name", "Cell Count"])

        def traverse(item):
            if not item.isHidden():
                rows.append([item.text(0), item.text(1), item.text(2)])
                for i in range(item.childCount()):
                    traverse(item.child(i))
        for i in range(self.tree_widget.topLevelItemCount()):
            traverse(self.tree_widget.topLevelItem(i))
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        logger.info("Exported table to %s", file_path)

    def on_item_clicked(self, item, column):
        selected_acronym = item.data(0, Qt.UserRole + 1)
        if not selected_acronym:
            return
        try:
            selected_id = int(item.data(0, Qt.UserRole))
        except (ValueError, TypeError):
            return
        descendant_acrs = self.bg_atlas.get_structure_descendants(selected_id)
        mapping = create_region_mapping(self.bg_atlas)
        id_to_acronym = {k: v[0] for k, v in mapping.items()}
        descendant_acrs = [id_to_acronym.get(
            did) for did in descendant_acrs if id_to_acronym.get(did)]
        selected_ids = [selected_id]
        for acr in descendant_acrs:
            try:
                selected_ids.append(self.bg_atlas.structures[acr]['id'])
            except KeyError:
                continue
        self.anno_layer.set_selected_ids(selected_ids)

    def reset_filter(self):
        self.anno_layer.reset_filter()
        self.search_bar.clear()

        def reset_item(item):
            item.setHidden(False)
            for i in range(item.childCount()):
                reset_item(item.child(i))
        for i in range(self.tree_widget.topLevelItemCount()):
            reset_item(self.tree_widget.topLevelItem(i))

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Napari viewer for brain region cell counts")
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

# ------------------------------------------------------------------------------
# Main Application Setup
# ------------------------------------------------------------------------------


def main():
    args = parse_args()
    logger.info("Starting application with arguments: %s", args)

    atlas, signal_image, annotation_data, original_annotation = load_atlas_and_data(
        args.atlas, args.signal_zarr, args.anno_zarr
    )

    if os.path.exists(args.points_final_csv):
        logger.info(
            "Processed points CSV '%s' exists. Loading precomputed points.", args.points_final_csv)
        df_points = pd.read_csv(args.points_final_csv)
    else:
        df_points = process_points(args.points_csv, original_annotation, atlas)
        if not df_points.empty:
            df_points.to_csv(args.points_final_csv, index=False)
            logger.info("Processed points CSV saved as '%s'.",
                        args.points_final_csv)
        else:
            logger.warning("No valid points data loaded.")

    if os.path.exists(args.cell_counts_csv):
        logger.info(
            "Cell counts CSV '%s' exists. Loading precomputed counts.", args.cell_counts_csv)
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

    # Precompute hierarchical counts with caching for fast lookup
    hierarchical_counts = precompute_hierarchical_counts(atlas, group_counts)

    viewer = napari.Viewer()
    viewer.add_image(signal_image, name="Anatomical Reference",
                     colormap="gray", contrast_limits=(0, 8000))
    anno_layer = FilteredLabels(
        annotation_data, name="Filtered Annotation", opacity=0.5)
    viewer.add_layer(anno_layer)

    region_tree = RegionTreeWidget(
        anno_layer, atlas.structures.tree, atlas, hierarchical_counts)
    viewer.window.add_dock_widget(
        region_tree, name="Brain Structure Tree", area="right")

    overlay_label = create_overlay_label(viewer)
    setup_mouse_move_callback(
        viewer, anno_layer, overlay_label, atlas, group_counts)
    add_points_layer(viewer, df_points)

    logger.info("Starting Napari event loop.")
    napari.run()


if __name__ == "__main__":
    main()
