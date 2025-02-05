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


# Function to normalize shapes and convert to bbox
def normalize_shapes_and_get_bboxes(shapes, img_width, img_height):
    normalized_bboxes = []

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
)
def run_countgd_widget(
    viewer: Viewer,
    label_type: str,
    text_prompt: str,
):
    """
    Plugin to run CountGD on the visible portion of the active image.

    Features:
      - Extracts the visible region of the active image (using camera center, zoom, and canvas size).
      - Allows users to optionally draw one or more exemplar bounding boxes on the image.
        When the "Draw Exemplar BBox" button is clicked, the shapes layer enters drawing mode.
        Users can draw and adjust rectangles; their normalized coordinates are computed and stored.
      - The counting function is called using the visible image and the provided caption.
        If exemplars are drawn, they are used; otherwise, the count runs using text only.
      - Detected points or bounding boxes are overlaid on the image.
    """
    global label_layer, current_label_type, exemplar_shapes_layer
    current_label_type = label_type
    # Will hold a list of normalized exemplar boxes (or remain None)
    exemplar_points = None

    # Ensure an image layer is selected.
    if not viewer.layers.selection:
        return "No image layer selected in Napari."
    image_layer_select = viewer.layers.selection.active
    if not isinstance(image_layer_select, Image):
        print(type(image_layer_select), 'Selected layer is not an image layer.')
        return "Selected layer is not an image layer."

    try:
        visible_image = image_layer_select.data

        # Access the shapes layer by name
        shapes_layer = viewer.layers['Shapes']
        exemplar_points = normalize_shapes_and_get_bboxes(
            shapes_layer.data, visible_image.shape[1], visible_image.shape[0])

        # --- Call the Counting Function ---
        labels, label_type_returned = generate_countgd_labels(
            visible_image,
            label_type=current_label_type,
            text_prompt=text_prompt,
            exemplar_image=None,  # Pass PIL Image
            # May be None if no boxes are drawn.
            exemplar_points=exemplar_points
        )
        # --- Display the Generated Labels ---
        if label_type_returned == 'points':
            # For a 2D image, offset returned points by the visible region's top-left.
            points = labels[:, :2]
            if label_layer is not None and isinstance(label_layer, Points):
                viewer.layers.remove(label_layer)
            label_layer = viewer.add_points(
                data=points,
                name='CountGD Points',
                size=5,
                face_color='red',
                edge_color='black'
            )
        elif label_type_returned == 'bboxes':
            if label_layer is not None and isinstance(label_layer, Shapes):
                viewer.layers.remove(label_layer)
            bbox_shapes_data = []
            for bbox in labels:
                min_b, max_b = bbox
                bbox_shapes_data.append([[min_b[0], max_b[0]], [min_b[1], max_b[0]], [
                                        min_b[1], max_b[1]], [min_b[0], max_b[1]]])
            label_layer = viewer.add_shapes(
                bbox_shapes_data,
                shape_type='rectangle',
                name='CountGD BBoxes',
                edge_color='cyan',
                face_color='transparent',
            )
        else:
            raise ValueError("Invalid label type returned from CountGD.")

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
