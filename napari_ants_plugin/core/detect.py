import argparse
import numpy as np
import zarr
import pandas as pd
from tqdm import tqdm
from napari_ants_plugin.gui.widgets import crop_shapes_from_image
from napari_ants_plugin.core.cells import remove_duplicate_cells
from napari_ants_plugin.core.labeling import generate_countgd_labels
from datetime import datetime
from napari_ants_plugin.gui.widgets import normalize_shapes_and_get_bboxes

# Global constant for large image tiling threshold.
LARGE_IMAGE_SIZE = (100, 100, 100)


def run_countgd(image_path,
                shapes=None,
                label_type="points",
                text_prompt="cell",
                confidence_threshold=0.01,
                current_z_slice_only=False,
                current_z=0,
                cell_size_radius=3.0,
                min_clip_value=0,
                max_clip_value=8000,
                csv_output_path=None,
                tile_size=(1024, 1024),
                overlap=(16, 16),
                ):
    """
    Run the CountGD detection algorithm on the provided image.

    Depending on the size of the image (and if it is 3D), the image is processed either by tiling or as a whole.
    Exemplar images are generated from the provided shapes.

    Args:
        image_path: The zarr input path.
        shapes (list or np.ndarray, optional): Shapes to use for exemplar generation.
        label_type (str): Either "points" or "bboxes".
        text_prompt (str): Object caption to pass to the detection.
        confidence_threshold (float): Confidence threshold for detection.
        current_z_slice_only (bool): For 3D images, whether to process only the specified z-slice.
        current_z (int): The current z-slice to process if current_z_slice_only is True.
        cell_size_radius (float): Radius parameter for duplicate cell removal.
        min_clip_value (int): Minimum clipping value for intensity.
        max_clip_value (int): Maximum clipping value for intensity.
        csv_output_path (str or None): If provided, save detected cells as CSV.

    Returns:
        dict: Dictionary containing the results. For example:
              { "label_type": "points", "detected_cells": np.ndarray } 
              or if bounding boxes: { "label_type": "bboxes", "bboxes": list }.
    """
    detected_cells = []
    # Will be updated based on generate_countgd_labels
    label_type_returned = label_type
    image_zarr = zarr.open(image_path)
    image = image_zarr["data"]

    # For images with an associated contrast range, you might compute these from the image;
    # here we simply use the provided min_clip_value and max_clip_value.
    img_shape = image.shape
    exemplar_points_from_shapes = []
    if shapes is not None:
        # Assume shapes is a list/array of 2D arrays (each with shape coordinates).
        # For 2D images, the shape coordinates should be in (y, x); for 3D, (z, y, x).
        if image.ndim == 2:
            exemplar_points_from_shapes = normalize_shapes_and_get_bboxes(
                shapes, img_width=img_shape[1], img_height=img_shape[0])
        elif image.ndim == 3:
            # For 3D, we use the first two dimensions for normalization.
            exemplar_points_from_shapes = normalize_shapes_and_get_bboxes(
                shapes, img_width=img_shape[2], img_height=img_shape[1])
    # Generate exemplar images from the provided shapes.
    cropped_exemplar_imgs = crop_shapes_from_image(image, shapes,
                                                   min_clip_value=min_clip_value,
                                                   max_clip_value=max_clip_value)

    # Decide whether to process by tiling (for large 3D images) or as a whole.
    if image.ndim == 3 and all(dim > large for dim, large in zip(image.shape, LARGE_IMAGE_SIZE)):
        # Tiling branch
        z_slices, height, width = image.shape
        stride = (tile_size[0] - overlap[0], tile_size[1] - overlap[1])

        for z in tqdm(range(z_slices)):
            if current_z_slice_only and z != current_z:
                continue
            slice_data = image[z]
            for y in range(0, height, stride[0]):
                for x in range(0, width, stride[1]):
                    y_end = min(y + tile_size[0], height)
                    x_end = min(x + tile_size[1], width)
                    tile = slice_data[y:y_end, x:x_end]

                    desired_height, desired_width = tile_size
                    if tile.shape[0] < desired_height or tile.shape[1] < desired_width:
                        pad_height = max(0, desired_height - tile.shape[0])
                        pad_width = max(0, desired_width - tile.shape[1])
                        tile = np.pad(
                            tile, ((0, pad_height), (0, pad_width)), mode='constant')

                    processed_tile = tile.clip(min_clip_value, max_clip_value)

                    if processed_tile.max() > 0:
                        processed_tile = processed_tile / processed_tile.max()
                    processed_tile = processed_tile * 255

                    # Overlay cropped exemplar images onto the tile.
                    tile_exemplar_points = []
                    current_x = 0
                    current_y = 0
                    row_max_height = 0
                    tile_exemplar_boxes = []
                    tile_h, tile_w = tile_size

                    for cei in cropped_exemplar_imgs:
                        cei = np.array(cei)
                        overlay_h, overlay_w = cei.shape[:2]
                        if current_x + overlay_w > tile_w:
                            current_x = 0
                            current_y += row_max_height
                            row_max_height = 0
                        if current_y + overlay_h > tile_h:
                            break
                        # Overlay exemplar image (here we simply overwrite the tile region).
                        processed_tile[current_y:current_y+overlay_h,
                                       current_x:current_x+overlay_w] = cei
                        overlay_box_global = [x+current_x, y+current_y,
                                              x+current_x+overlay_w, y+current_y+overlay_h]
                        tile_exemplar_boxes.append(overlay_box_global)
                        # Normalize overlay box coordinates relative to tile width.
                        overlay_box_norm = np.array(
                            [current_x, current_y, current_x +
                                overlay_w, current_y+overlay_h],
                            dtype=float) / tile_w
                        tile_exemplar_points.append(overlay_box_norm.tolist())
                        current_x += overlay_w
                        row_max_height = max(row_max_height, overlay_h)

                    # Call the counting function for this tile.
                    labels, label_type_returned = generate_countgd_labels(
                        processed_tile,
                        label_type=label_type,
                        text_prompt=text_prompt,
                        exemplar_image=None,
                        exemplar_points=tile_exemplar_points,
                        offset_x=x,
                        offset_y=y,
                        z_slice=z,
                        confidence_threshold=confidence_threshold
                    )

                    # Filter out any returned points that match an exemplar overlay.
                    for cell in labels:
                        cell_z, cell_y, cell_x = cell
                        inside_box = False
                        for global_box in tile_exemplar_boxes:
                            if (cell_x >= global_box[0] and cell_x <= global_box[2] and
                                    cell_y >= global_box[1] and cell_y <= global_box[3]):
                                inside_box = True
                        if not inside_box:
                            detected_cells.append(cell)
            print(
                f"Detected {len(detected_cells)} cells from slice {0} to {z}")
        if detected_cells:
            detected_cells = list(set(tuple(cell) for cell in detected_cells))
            detected_cells = remove_duplicate_cells(
                detected_cells, cell_size_radius=cell_size_radius)
            points = np.array(detected_cells)
            points = np.unique(points, axis=0)
            print("Total number of cells detected:", len(points))
            detected_cells = points.tolist()

    else:
        # Non-tiling branch: process the entire image.
        labels, label_type_returned = generate_countgd_labels(
            image,
            label_type=label_type,
            text_prompt=text_prompt,
            exemplar_image=None,
            exemplar_points=exemplar_points_from_shapes,
            confidence_threshold=confidence_threshold,
        )
        if label_type_returned == 'points':
            # Assume labels is an array whose first two columns are the (y,x) coordinates.
            points = labels[:, :2]
            detected_cells = points.tolist()
        elif label_type_returned == 'bboxes':
            bbox_shapes_data = []
            for bbox in labels:
                min_b, max_b = bbox
                bbox_shapes_data.append([
                    [min_b[0], max_b[0]],
                    [min_b[1], max_b[0]],
                    [min_b[1], max_b[1]],
                    [min_b[0], max_b[1]]
                ])
            detected_cells = bbox_shapes_data
        else:
            raise ValueError("Invalid label type returned from CountGD.")

    # Optionally, save detected cells to CSV if a path is provided.
    if detected_cells and csv_output_path:
        detected_array = np.array(detected_cells)
        num_dims = detected_array.shape[1]
        if num_dims == 2:
            columns = ["y", "x"]
        elif num_dims == 3:
            columns = ["z", "y", "x"]
        else:
            columns = [f"dim_{i}" for i in range(num_dims)]
        df = pd.DataFrame(detected_array, columns=columns)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = csv_output_path if csv_output_path.endswith(
            ".csv") else f"detected_cells_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Unique detected cells saved to {csv_filename}")

    return {"label_type": label_type_returned, "detected_cells": detected_cells}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CountGD without Napari")
    parser.add_argument("--image_path", type=str,
                        default="output_pipeline/intermediate/signal_image.zarr",
                        help="Path to the input image")
    parser.add_argument("--label_type", type=str,
                        choices=['points', 'bboxes'], default='points', help="Label type")
    parser.add_argument("--text_prompt", type=str,
                        default="cell", help="Object caption")
    parser.add_argument("--confidence_threshold", type=float,
                        default=0.01, help="Confidence threshold")
    parser.add_argument("--current_z_slice_only",
                        action='store_true', help="Process only first Z-slice")
    parser.add_argument("--cell_size_radius", type=float,
                        default=3.0, help="Cell size radius")

    args = parser.parse_args()
    run_countgd(
        args.image_path,
        shapes=None,
        label_type=args.label_type,
        text_prompt=args.text_prompt,
        confidence_threshold=args.confidence_threshold,
        current_z_slice_only=args.current_z_slice_only,
        cell_size_radius=args.cell_size_radius,
    )
