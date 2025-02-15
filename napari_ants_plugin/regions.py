import os
import numpy as np
import pandas as pd
import napari
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas

# Qt imports
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, QLabel
from qtpy.QtGui import QFont, QCursor
from qtpy.QtCore import Qt

# Napari imports
from napari.layers import Labels


# ------------------------------------------------------------------------------
# Custom Labels Layer
# ------------------------------------------------------------------------------
class FilteredLabels(Labels):
    """
    Custom Labels layer that shows a filtered version of the annotation data.
    When a set of region IDs is selected, only those regions are visible.
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._original_data = data
        self._selected_ids = None          # list of region IDs to show

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
def load_atlas_and_data():
    """
    Load the BrainGlobeAtlas as well as the anatomical image and annotation data.
    Returns:
        tuple: (atlas, signal_image, annotation_data, original_annotation)
    """
    atlas = BrainGlobeAtlas("kim_mouse_25um", check_latest=False)

    # Load anatomical image data (Zarr format)
    signal_zarr = zarr.open("MF1wt_125F_W_BS_640.zarr", mode="r")
    signal_image = signal_zarr["data"]

    # Load annotation data (Zarr format)
    anno_zarr = zarr.open("resampled_full_anno.zarr", mode="r")
    annotation_data = anno_zarr["data"]
    original_annotation = annotation_data

    return atlas, signal_image, annotation_data, original_annotation


def create_region_mapping(bg_atlas: BrainGlobeAtlas) -> dict:
    """
    Build a mapping from region id to a tuple (acronym, name).
    """
    return {
        int(row["id"]): (row["acronym"], row["name"])
        for _, row in bg_atlas.lookup_df.iterrows()
    }


def add_region_info_to_points(df: pd.DataFrame, annotation: np.ndarray, bg_atlas: BrainGlobeAtlas) -> pd.DataFrame:
    """
    Given a DataFrame of points (with x, y, z columns), add region info (acronym and name)
    using the annotation array and atlas lookup. Points out-of-bounds are filtered out.
    """
    x = df["x"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    z = df["z"].to_numpy(dtype=int)
    shape = annotation.shape  # Expected order: (Z, Y, X)

    valid_mask = (z >= 0) & (z < shape[0]) & (y >= 0) & (
        y < shape[1]) & (x >= 0) & (x < shape[2])
    if not np.all(valid_mask):
        print(f"Filtered out {np.sum(~valid_mask)} out-of-bound points.")
    df = df.loc[valid_mask].copy()
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # Get region IDs from annotation and map to (acronym, name)
    region_vals = annotation[z, y, x]
    mapping = create_region_mapping(bg_atlas)
    region_info = pd.Series(region_vals).map(mapping)
    df["region_acronym"] = region_info.apply(
        lambda x: x[0] if isinstance(x, tuple) else None)
    df["region_name"] = region_info.apply(
        lambda x: x[1] if isinstance(x, tuple) else None)
    return df


def process_points(csv_path: str, annotation: np.ndarray, bg_atlas: BrainGlobeAtlas) -> pd.DataFrame:
    """
    Load and process the points CSV:
      - Add region acronym and name for each point.
      - Filter out points with invalid or unknown region info.
    Returns:
        pd.DataFrame: Processed points data.
    """
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Skipping point processing.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print("Loaded Points CSV:")
    print(df.head())
    df = add_region_info_to_points(df, annotation, bg_atlas)
    df_filtered = df[df["region_acronym"].notna() & (
        df["region_acronym"] != "Unknown")].copy()
    return df_filtered


# ------------------------------------------------------------------------------
# GroupBy-based Hierarchical Counting Functions
# ------------------------------------------------------------------------------
def count_points_in_region_hierarchical(group_counts: dict, region_acronym: str, atlas: BrainGlobeAtlas) -> int:
    """
    Given a precomputed dictionary of counts for each region (group_counts),
    return the total count for the specified region and all its descendant regions.
    """
    if region_acronym is None:
        return 0
    descendant_acrs = atlas.get_structure_descendants(region_acronym)
    total = group_counts.get(region_acronym, 0)
    for acr in descendant_acrs:
        total += group_counts.get(acr, 0)
    return total


def save_cell_counts_by_region(df: pd.DataFrame, atlas: BrainGlobeAtlas, output_file: str = "cell_counts_by_region.csv"):
    """
    Save a CSV file with cell counts for every brain region (including descendants)
    by leveraging a precomputed groupby result.
    """
    # Precompute counts per region using groupby
    group_counts = df.groupby("region_acronym").size().to_dict()
    all_counts = []
    for _, row in atlas.lookup_df.iterrows():
        acr = row["acronym"]
        count = count_points_in_region_hierarchical(group_counts, acr, atlas)
        all_counts.append({
            "acronym": acr,
            "name": row["name"],
            "cell_count": count
        })
    counts_df = pd.DataFrame(all_counts)
    counts_df.to_csv(output_file, index=False)
    print(f"Saved cell counts for each region to '{output_file}'.")


# ------------------------------------------------------------------------------
# UI Helper Functions
# ------------------------------------------------------------------------------
def get_structure_info(structure_id, atlas: BrainGlobeAtlas):
    """
    Return the (acronym, name) for the given structure ID using the atlas lookup.
    """
    try:
        structure_id = int(structure_id)
    except (ValueError, TypeError):
        return (None, None)
    mapping = create_region_mapping(atlas)
    return mapping.get(structure_id, (None, None))


def add_points_layer(viewer, df: pd.DataFrame):
    """
    Add a Napari Points layer from the processed DataFrame.
    Expects the DataFrame to have columns 'x', 'y', 'z' (converted to (z, y, x)).
    """
    if df.empty:
        return
    points = df[['z', 'y', 'x']].to_numpy()
    viewer.add_points(points, name="Points", size=10,
                      face_color="red", border_color="white")


def in_layer(layer, world_coord):
    """
    Check whether the world coordinate falls within the bounds of the given layer.
    """
    data_coord = layer.world_to_data(world_coord)
    data_idx = np.round(data_coord).astype(int)
    if np.any(data_idx < 0) or np.any(data_idx >= np.array(layer.data.shape)):
        return False
    return True


def create_overlay_label(viewer) -> QLabel:
    """
    Create an overlay label that displays structure information.
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


def setup_mouse_move_callback(viewer, anno_layer, label, atlas, group_counts):
    """
    Attach a mouse move callback to update the overlay label with information
    about the brain structure under the cursor using precomputed group counts.
    """
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
# Region Tree Widget for Filtering the Annotation Layer
# ------------------------------------------------------------------------------
class RegionTreeWidget(QWidget):
    def __init__(self, anno_layer, bg_tree, bg_atlas):
        super().__init__()
        self.anno_layer = anno_layer
        self.bg_tree = bg_tree
        self.bg_atlas = bg_atlas
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Brain Region"])
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
        self.populate_tree()
        self.tree_widget.itemClicked.connect(self.on_item_clicked)

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

        layout.addWidget(self.tree_widget)
        layout.addWidget(self.reset_button)
        self.setLayout(layout)

    def populate_tree(self):
        root_id = self.bg_tree.root
        root_node = self.bg_tree.get_node(root_id)
        root_item = QTreeWidgetItem([str(root_node.tag)])
        region_id = root_node.data.get("id", "") if root_node.data else ""
        root_item.setData(0, Qt.UserRole, region_id)
        self.tree_widget.addTopLevelItem(root_item)
        self.add_children(root_item, root_node)
        self.tree_widget.expandAll()

    def add_children(self, parent_item, parent_node):
        for child in self.bg_tree.children(parent_node.identifier):
            child_item = QTreeWidgetItem([str(child.tag)])
            region_id = child.data.get("id", "") if child.data else ""
            child_item.setData(0, Qt.UserRole, region_id)
            parent_item.addChild(child_item)
            self.add_children(child_item, child)

    def on_item_clicked(self, item, column):
        selected_acronym = item.text(0)
        if not selected_acronym:
            return
        try:
            selected_id = int(item.data(0, Qt.UserRole))
        except (ValueError, TypeError):
            return

        descendant_acrs = self.bg_atlas.get_structure_descendants(
            selected_acronym)
        selected_ids = [selected_id]
        for acr in descendant_acrs:
            try:
                selected_ids.append(self.bg_atlas.structures[acr]['id'])
            except KeyError:
                continue
        self.anno_layer.set_selected_ids(selected_ids)

    def reset_filter(self):
        self.anno_layer.reset_filter()


# ------------------------------------------------------------------------------
# Main Application Setup
# ------------------------------------------------------------------------------
def main():
    # Load atlas, anatomical image, and annotation data.
    atlas, signal_image, annotation_data, original_annotation = load_atlas_and_data()

    # Process points CSV.
    points_csv = "points.csv"
    df_points = process_points(points_csv, original_annotation, atlas)
    if not df_points.empty:
        processed_csv = "points_final.csv"
        df_points.to_csv(processed_csv, index=False)
        print(f"Processed points CSV saved as '{processed_csv}'.")
        # Save cell counts per region using the efficient groupby approach.
        save_cell_counts_by_region(df_points, atlas)
        # Precompute groupby counts for efficient hierarchical queries.
        group_counts = df_points.groupby("region_acronym").size().to_dict()
    else:
        group_counts = {}

    # Create the Napari viewer and add the anatomical image.
    viewer = napari.Viewer()
    viewer.add_image(signal_image, name="Anatomical Reference",
                     colormap="gray", contrast_limits=(0, 8000))

    # Add the custom FilteredLabels layer.
    anno_layer = FilteredLabels(
        annotation_data, name="Filtered Annotation", opacity=0.5)
    viewer.add_layer(anno_layer)

    # Add the region tree widget for filtering.
    region_tree = RegionTreeWidget(anno_layer, atlas.structures.tree, atlas)
    viewer.window.add_dock_widget(
        region_tree, name="Brain Structure Tree", area="right")

    # Create an overlay label and set up the mouse move callback using the precomputed group_counts.
    overlay_label = create_overlay_label(viewer)
    setup_mouse_move_callback(
        viewer, anno_layer, overlay_label, atlas, group_counts)

    # Add the processed points as a layer.
    add_points_layer(viewer, df_points)

    napari.run()


if __name__ == "__main__":
    main()
