from magicgui import magic_factory, widgets
from magicgui.widgets import PushButton, Image as ImageWidget
from napari.layers import Image, Shapes, Points
from napari.types import LayerDataTuple
from napari import Viewer
from skimage.io import imread, imsave
import os
import numpy as np
import json

# Make sure these are correctly implemented and accessible
from ..core.align import align_images  # noqa: F401 (if used elsewhere)
from ..core.transform import transform_image  # noqa: F401 (if used elsewhere)
from ..core.utils import log_score  # noqa: F401 (if used elsewhere)
from ..core.labeling import generate_countgd_labels  # noqa: F401 (if used - important for CountGD)

# Globals
label_layer = None
current_label_type = None
exemplar_shapes_layer = None  # Stores the shapes layer for exemplar bbox drawing
large_image_size  = (100,100,100)



def crop_shapes_from_image(image, labeled_shapes,min_clip_value=0,max_clip_value=8000):
    """
    Crops regions from an image based on labeled shapes.
    Coordinates in labeled_shapes are assumed to be in Z, Y, X format.

    Args:
        image (np.ndarray): The large image to crop from (assuming it's a NumPy array, could be 2D or 3D).
        labeled_shapes (list of np.ndarray): A list where each np.ndarray represents a shape.
                                             Each shape array is expected to be in the format:
                                             [[z1, y1, x1], [z2, y2, x2], ..., [zn, yn, xn]].

    Returns:
        list of np.ndarray: A list of cropped image regions, one for each shape.
                            Returns an empty list if there are no shapes or an error occurs.
    """
    cropped_images = []
    if not labeled_shapes:
        print("No shapes provided to crop.")
        return cropped_images

    try:
        for shape in labeled_shapes:
            # Extract z, y, and x coordinates from the shape (Z, Y, X format)
            z_coords = shape[:, 0]
            y_coords = shape[:, 1]
            x_coords = shape[:, 2]

            # Get bounding box coordinates (min and max) for x and y
            xmin = int(np.min(x_coords))
            ymin = int(np.min(y_coords))
            xmax = int(np.max(x_coords))
            ymax = int(np.max(y_coords))

            # Determine z-slice to use (using min z for simplicity)
            z_slice = int(np.min(z_coords))

            # Handle potential 2D or 3D image input
            if image.ndim == 2:
                cropped_image = image[ymin:ymax, xmin:xmax]
            elif image.ndim == 3:
                # Assuming the image is (Z, Y, X) or similar - adjust if your image dimension order is different
                if z_slice < image.shape[0]: # Check if z_slice is within image bounds
                    cropped_image = image[z_slice, ymin:ymax, xmin:xmax]
                else:
                    print(f"Warning: z_slice {z_slice} is out of bounds for image with shape {image.shape}. Skipping crop for this shape.")
                    continue # Skip to the next shape if z_slice is invalid
            else:
                print(f"Error: Image has unsupported dimensions ({image.ndim}). Expecting 2D or 3D.")
                return [] # Return empty list if image dimensions are wrong

            cropped_image.clip(min_clip_value,max_clip_value)
            cropped_image = cropped_image / cropped_image.max()
            cropped_image = cropped_image * 255
            cropped_images.append(cropped_image)

    except Exception as e:
        print(f"Error during cropping: {e}")
        return [] # Return empty list if any error occurs during processing

    return cropped_images


# Function to normalize shapes and convert to bbox
def normalize_shapes_and_get_bboxes(shapes, img_width, img_height):
    normalized_bboxes = []
    print("labeled shapes :", shapes)

    for shape in shapes:
        # Extract x and y coordinates from the shape
        # X is the second column (since Napari uses (Y, X) format)
        x_coords = shape[:, 1]
        y_coords = shape[:, 0]  # Y is the first column

        # Get bounding box coordinates (min and max)
        xmin = np.min(x_coords) / img_width
        ymin = np.min(y_coords) / img_height
        xmax = np.max(x_coords) / img_width
        ymax = np.max(y_coords) / img_height

        # Append normalized bbox [xmin, ymin, xmax, ymax]
        normalized_bboxes.append([xmin, ymin, xmax, ymax])

    return normalized_bboxes


@magic_factory(
    call_button="Run CountGD",
    label_type={"choices": ['points', 'bboxes'], "label": "Label Type"},
    text_prompt={"label": "Object Caption", "value": "cell"},
    confidence_threshold={"label":"Confidence Threshold", "value": 0.23}
)
def run_countgd_widget(
    viewer: Viewer,
    label_type: str,
    text_prompt: str,
    confidence_threshold: float = 0.23,
):
    """
    Plugin to run CountGD on the visible portion of the active image.

    Features:
      - Extracts the visible region of the active image (using camera center, zoom, and canvas size).
      - Allows users to optionally draw one or more exemplar bounding boxes on the image.
      - The counting function is called using the visible image and the provided caption.
      - If exemplar boxes (or points) were added, those points are filtered out of the detected cells.
      - Detected points (with duplicates removed) are saved to a CSV file.
    """
    global label_layer, current_label_type, exemplar_shapes_layer
    current_label_type = label_type
    # This will hold the detected cell coordinates (points)
    detected_cells = []

    zoom_level = viewer.camera.zoom
    print(f"Current zoom level: {zoom_level}")

    # Ensure an image layer is selected.
    if not viewer.layers.selection:
        return "No image layer selected in Napari."
    image_layer_select = viewer.layers.selection.active
    if not isinstance(image_layer_select, Image):
        print(type(image_layer_select), 'Selected layer is not an image layer.')
        return "Selected layer is not an image layer."

    try:
        visible_image = image_layer_select.data
        print(f"Original image shape: {visible_image.shape}")
        transform_shape = image_layer_select.extent.world[1]
        print(f"Displayed Image size in World coordinates: {transform_shape}")
        contrast_limits = image_layer_select.contrast_limits
        print(f"Contrast Limits: {contrast_limits}")
        min_clip_value,max_clip_value = contrast_limits

        cursor_position = viewer.cursor.position 
        intensity_value = image_layer_select.get_value(cursor_position)
        print(f"Intensity Value at Cursor Position: {intensity_value}")
        print(f"Cursor Position: {cursor_position}")

        # Get the exemplar shapes from the "Shapes" layer.
        shapes_layer = viewer.layers['Shapes']
        exemplar_points_from_shapes = normalize_shapes_and_get_bboxes(
            shapes_layer.data, visible_image.shape[1], visible_image.shape[0])
        cropped_examplar_imgs = crop_shapes_from_image(visible_image, 
                                                       shapes_layer.data,
                                                       min_clip_value=min_clip_value,
                                                       max_clip_value=max_clip_value
                                                       )
        
        # Decide whether to process the image by tiling or as a whole.
        if all(img_dim > large_dim for img_dim, large_dim in zip(visible_image.shape, large_image_size)):
            # --- Tiling branch ---
            z_slices, height, width = visible_image.shape
            camera_center = viewer.camera.center
            print(f"Center of the camera: {camera_center}")

            tile_size = (1024, 1024)  # (height, width)
            overlap = (16, 16)        # (height, width)
            stride = (tile_size[0] - overlap[0], tile_size[1] - overlap[1])
            
            for z in range(1051,z_slices):
                slice_data = visible_image[z]
                print(slice_data.shape, "Current slice number:", z)
                # Loop over tiles
                for y in range(0, height, stride[0]):
                    for x in range(0, width, stride[1]):
                        y_end = min(y + tile_size[0], height)
                        x_end = min(x + tile_size[1], width)
                        tile = slice_data[y:y_end, x:x_end]
                        processed_tile = tile.clip(min_clip_value, max_clip_value)
                        if processed_tile.max() > 0:
                            processed_tile = processed_tile / processed_tile.max()
                        processed_tile = processed_tile * 255

                        # --- Overlay exemplar images onto this tile ---
                        tile_exemplar_points = []  # list for this tile
                        current_x = 0
                        current_y = 0
                        row_max_height = 0
                        tile_h, tile_w = tile_size
                        tile_exemplar_boxes = []

                        for cei in cropped_examplar_imgs:
                            if not isinstance(cei, np.ndarray):
                                cei = np.array(cei)
                            overlay_h, overlay_w = cei.shape[:2]

                            # If the exemplar does not fit in the current row, wrap to the next row.
                            if current_x + overlay_w > tile_w:
                                current_x = 0
                                current_y += row_max_height
                                row_max_height = 0

                            # Break out if no vertical space remains.
                            if current_y + overlay_h > tile_h:
                                break

                            # Overlay the exemplar image onto the processed tile.
                            processed_tile[current_y:current_y+overlay_h, current_x:current_x+overlay_w] = cei

                            # Compute normalized overlay box coordinates (x1,y1,x2,y2).
                            overlay_box = [current_x, current_y, current_x + overlay_w, current_y + overlay_h]
                            tile_exemplar_boxes.append(overlay_box)
                            overlay_box_norm = np.array(
                                overlay_box,
                                dtype=float
                            ) / tile_w
                            tile_exemplar_points.append(overlay_box_norm.tolist())
                            current_x += overlay_w
                            row_max_height = max(row_max_height, overlay_h)

                        # --- Call the counting function for this tile ---
                        labels, label_type_returned = generate_countgd_labels(
                            processed_tile,
                            label_type=current_label_type,
                            text_prompt=text_prompt,
                            exemplar_image=None if len(cropped_examplar_imgs) < 1 else cropped_examplar_imgs[0],
                            exemplar_points=tile_exemplar_points,
                            offset_x=x,
                            offset_y=y,
                            z_slice=z,
                            confidence_threshold=confidence_threshold
                        )
                        
                        # --- Filter out any returned points that match the exemplar overlay points ---
                        for cell in labels:
                            cell_z, cell_y, cell_x = cell
                            inside_box = False
                            for gloabl_box in tile_exemplar_boxes:
                                if (cell_x >= gloabl_box[0] and cell_x <= gloabl_box[2] and 
                                    cell_y >= gloabl_box[1] and cell_y <= gloabl_box[3]):
                                    inside_box = True
                            if not inside_box:
                                detected_cells.append(cell)
                break
            
            if len(detected_cells) > 0:
                points = np.array(detected_cells)
                print("Total number of cells detected:", len(detected_cells))
                # if label_layer is not None and isinstance(label_layer, Points):
                #     viewer.layers.remove(label_layer)
                label_layer = viewer.add_points(
                    data=points,
                    name='Cells',
                    size=10,
                    face_color='red',
                    edge_color='black'
                )
        else:
            # --- Non-tiling branch: process the entire image ---
            labels, label_type_returned = generate_countgd_labels(
                visible_image,
                label_type=current_label_type,
                text_prompt=text_prompt,
                exemplar_image=None,
                exemplar_points=exemplar_points_from_shapes,
                confidence_threshold=confidence_threshold,

            )

            if label_type_returned == 'points':
                points = labels[:, :2]
                label_layer = viewer.add_points(
                    data=points,
                    name='Cells',
                    size=10,
                    face_color='red',
                    edge_color='black'
                )
                # Save these points into detected_cells (for CSV export)
                detected_cells = points.tolist()
            elif label_type_returned == 'bboxes':
                if label_layer is not None and isinstance(label_layer, Shapes):
                    viewer.layers.remove(label_layer)
                bbox_shapes_data = []
                for bbox in labels:
                    min_b, max_b = bbox
                    bbox_shapes_data.append([
                        [min_b[0], max_b[0]],
                        [min_b[1], max_b[0]],
                        [min_b[1], max_b[1]],
                        [min_b[0], max_b[1]]
                    ])
                label_layer = viewer.add_shapes(
                    bbox_shapes_data,
                    shape_type='rectangle',
                    name='bboxes',
                    edge_color='cyan',
                    face_color='transparent',
                )
                # (Optionally, you might extract center points from bboxes if you wish to save them.)
            else:
                raise ValueError("Invalid label type returned from CountGD.")
        
        # --- Save unique detected cells to a CSV file ---
        if detected_cells:
            detected_array = np.array(detected_cells)
            # Remove duplicate points.
            unique_points = np.unique(detected_array, axis=0)
            # Choose column names based on the number of dimensions.
            num_dims = unique_points.shape[1]
            if num_dims == 2:
                columns = ["y", "x"]
            elif num_dims == 3:
                columns = ["z", "y", "x"]
            else:
                columns = [f"dim_{i}" for i in range(num_dims)]
            import pandas as pd
            df = pd.DataFrame(unique_points, columns=columns)
            csv_filename = "detected_cells.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Unique detected cells saved to {csv_filename}")
        
        return f"CountGD labels generated ({label_type_returned})."
    
    except Exception as e:
        print(f"CountGD Error: {e}")
        return f"Error running CountGD: {e}"



@magic_factory(
    call_button="Save ROI Image",
    output_file_path={"widget_type": "FileEdit", "mode": "save",
                      "label": "Save ROI Image As", "filter": "*.tif"}
)
def save_roi_image_widget(viewer: Viewer, output_file_path: str):
    """Widget to save the ROI image as a TIFF file from the currently selected image layer."""
    global roi_shape_layer

    if not viewer.layers.selection:
        return "No image layer selected in Napari."

    image_layer_select = viewer.layers.selection.active

    if not isinstance(image_layer_select, Image):
        return "Selected layer is not an image layer."

    if output_file_path == "":
        return "Please specify a file path to save the ROI image."

    # Use the ROI from the ROI Region layer if available; otherwise use the whole image.
    roi_coords = None
    if roi_shape_layer is not None and roi_shape_layer.data:
        roi_coords = roi_shape_layer.data[0]
    if roi_coords is None:
        print("No ROI shape data found. Will use the whole image.")
        ndim = image_layer_select.data.ndim
        roi_coords = np.array(
            [[0] * ndim, list(image_layer_select.data.shape)])

    try:
        # Determine the bounding coordinates.
        min_coords = np.min(roi_coords, axis=0).astype(int)
        max_coords = np.max(roi_coords, axis=0).astype(int)

        # Clip coordinates so they lie within image bounds.
        img_shape = np.array(image_layer_select.data.shape)
        min_coords = np.clip(min_coords, 0, img_shape)
        max_coords = np.clip(max_coords, 0, img_shape)

        # Create slices for each dimension.
        slices = tuple(slice(m, M) for m, M in zip(min_coords, max_coords))
        roi_image_to_save = image_layer_select.data[slices]

        imsave(output_file_path, roi_image_to_save)
        return f"ROI image saved to: {output_file_path}"

    except Exception as e:
        print(f"Save ROI Image Error: {e}")
        return f"Error saving ROI image: {e}"


@magic_factory(
    call_button="Save Labels",
    output_file_path={"widget_type": "FileEdit", "mode": "save",
                      "label": "Save Labels As", "filter": "*.csv *.json"}
)
def save_labels_widget(viewer: Viewer, output_file_path: str):
    """Widget to save the generated labels."""
    global label_layer, current_label_type

    if label_layer is None:
        return "Run CountGD to generate labels first."

    if output_file_path == "":
        return "Please specify a file path to save labels."

    try:
        if current_label_type == 'points':
            # Save point labels as CSV.
            labels_data = label_layer.data
            header = "x,y,z"
            np.savetxt(output_file_path, labels_data,
                       delimiter=',', header=header, comments='')
        elif current_label_type == 'bboxes':
            # Save bounding boxes as JSON.
            labels_data = label_layer.data
            json_compatible_bboxes = [
                [vertex.tolist() for vertex in bbox] for bbox in labels_data
            ]
            with open(output_file_path, 'w') as f:
                json.dump({"label_type": "bboxes",
                          "bboxes": json_compatible_bboxes}, f, indent=2)
        else:
            raise ValueError("Unknown label type for saving.")

        return f"Labels saved to: {output_file_path}"

    except Exception as e:
        print(f"Save Labels Error: {e}")
        return f"Error saving labels: {e}"


@magic_factory(call_button="Align Images",
               fixed_image_path={"widget_type": "FileEdit",
                                 "label": "Fixed Image File"},
               moving_image_path={"widget_type": "FileEdit", "label": "Moving Image File"})
def image_alignment_widget(viewer: Viewer,
                           fixed_image_path: str = '',
                           moving_image_path: str = '',
                           transform_type: str = 'SyNRA') -> LayerDataTuple:
    """Widget for image alignment using ANTs."""

    # Verify that both image paths are provided
    if not fixed_image_path or not moving_image_path:
        raise ValueError("Please select both fixed and moving image files.")

    # Check if files exist
    if not os.path.exists(fixed_image_path):
        raise FileNotFoundError(
            f"Fixed image file not found: {fixed_image_path}")
    if not os.path.exists(moving_image_path):
        raise FileNotFoundError(
            f"Moving image file not found: {moving_image_path}")

    # Load the images from the file paths
    fixed_image_data = imread(fixed_image_path)
    moving_image_data = imread(moving_image_path)

    # Add images to the viewer only if they are not already added
    if "Fixed Image" not in viewer.layers:
        viewer.add_image(fixed_image_data, name="Fixed Image")
    if "Moving Image" not in viewer.layers:
        viewer.add_image(moving_image_data, name="Moving Image")

    # Perform alignment
    aligned_image = align_images(
        fixed_image_data, moving_image_data, transform_type=transform_type)

    # Return the aligned image as a new layer
    return (aligned_image, {"name": "Aligned Image"})


@magic_factory(call_button="Transform Image",
               image_path={"widget_type": "FileEdit", "label": "Image File"},
               reference_image_path={"widget_type": "FileEdit", "label": "Reference Image File"})
def image_transformation_widget(viewer: Viewer,
                                image_path: str = '',
                                reference_image_path: str = '',
                                transform_dir: str = '',
                                invert: bool = False,
                                interpolation: str = 'bspline') -> LayerDataTuple:
    """Widget for image transformation using ANTs."""

    # Verify that both image paths are provided
    if not image_path or not reference_image_path:
        raise ValueError("Please select both image and reference image files.")

    # Check if files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(reference_image_path):
        raise FileNotFoundError(
            f"Reference image file not found: {reference_image_path}")

    # Load the images from the file paths
    image_data = imread(image_path)
    reference_image_data = imread(reference_image_path)

    # Add images to the viewer only if they are not already added
    if "Image" not in viewer.layers:
        viewer.add_image(image_data, name="Image")
    if "Reference Image" not in viewer.layers:
        viewer.add_image(reference_image_data, name="Reference Image")

    # Perform the transformation
    transformed_image = transform_image(
        image_data, reference_image_data, transform_dir, invert=invert, interpolation=interpolation)

    # Return the transformed image as a new layer
    return (transformed_image, {"name": "Transformed Image"})


@magic_factory(call_button="Submit Score",
               aligned_image_path={"widget_type": "FileEdit", "label": "Aligned Image Path"})
def scoring_widget(viewer: Viewer,
                   aligned_image_path: str = '',
                   score: int = 0,
                   comments: str = '') -> None:
    """Widget for scoring registration results."""

    # Check if the aligned image is already in the viewer
    aligned_image_layer = None
    for layer in viewer.layers:
        if layer.name == "Aligned Image":
            aligned_image_layer = layer
            break

    # If no aligned image is in the viewer, load from the provided path
    if aligned_image_layer is None:
        if not aligned_image_path:
            raise ValueError(
                "No 'Aligned Image' found in the viewer. Please provide a path to load one.")

        if not os.path.exists(aligned_image_path):
            raise FileNotFoundError(
                f"Aligned image file not found: {aligned_image_path}")

        # Load the aligned image from the specified path into the viewer
        aligned_image_data = imread(aligned_image_path)

        # Add the aligned image to the viewer as a new layer
        aligned_image_layer = viewer.add_image(
            aligned_image_data, name="Aligned Image")

    # Ensure the score is within the valid range
    if not (0 <= score <= 10):
        raise ValueError("Score must be between 0 and 10.")

    # Log the score and comments
    log_score(aligned_image_layer.name, score, comments)


# --- Napari Plugin Registration ---
def napari_widgets():
    return [run_countgd_widget, save_roi_image_widget, save_labels_widget,
            image_alignment_widget, image_transformation_widget, scoring_widget]
