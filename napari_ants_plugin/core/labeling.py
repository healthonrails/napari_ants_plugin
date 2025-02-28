import numpy as np
from PIL import Image
# Assuming predict.py is in the same directory
from napari_ants_plugin.core.countgd.predict import ObjectCounter
from napari.layers import Image as NapariImage  # To avoid naming conflict

# Initialize ObjectCounter (load model once for efficiency)
object_counter = ObjectCounter()
DEFAULT_TEXT_PROMPT = "cell"  # Or any default prompt you prefer


def generate_countgd_labels(image_roi, label_type='points',
                            text_prompt=DEFAULT_TEXT_PROMPT,
                            exemplar_image=None,
                            exemplar_points=None,
                            offset_x=0,
                            offset_y=0,
                            z_slice=None,
                            confidence_threshold=0.23,
                            ):
    """
    Generates CountGD labels (points or bounding boxes) using the ObjectCounter model.

    Args:
        image_roi (np.ndarray or NapariImage): 2D or 3D NumPy array or Napari Image layer representing the ROI.
        label_type (str): 'points' or 'bboxes'.
        text_prompt (str, optional): Textual prompt for object counting. Defaults to DEFAULT_TEXT_PROMPT.
        exemplar_image (np.ndarray or NapariImage or PIL.Image, optional): The exemplar image. Defaults to None.
        exemplar_points (list, optional): List of exemplar bounding boxes in normalized [xmin, ymin, xmax, ymax] format. Defaults to None.

    Returns:
        tuple: (labels, label_type)
               - labels:  If label_type is 'points': a NumPy array of point coordinates (N, 2) for 2D images
                          or (N, 3) for volumetric images.
                          If label_type is 'bboxes': a list of bounding box tuples [(min_coords, max_coords), ...].
               - label_type: 'points' or 'bboxes' (returned as a check).
    """
    # Print the ROI shape.
    # roi_shape = image_roi.shape if isinstance(
    #     image_roi, np.ndarray) else image_roi.data.shape
    # print("CountGD running on ROI shape:",
    #       roi_shape, f"with prompt: '{text_prompt}'")

    # Extract the NumPy array from a Napari Image layer, if necessary.
    image_roi_np = image_roi.data if isinstance(
        image_roi, NapariImage) else image_roi

    # Determine if we treat the input as a 2D color image.
    # A typical 2D color image has 3 (or 4) channels (e.g. shape: (height, width, 3) or (height, width, 4)).
    is_color_2d = (image_roi_np.ndim == 3 and image_roi_np.shape[-1] in [3, 4])

    # Convert the ROI to a PIL Image.
    if is_color_2d:
        if image_roi_np.shape[-1] == 4:
            image_pil = Image.fromarray(image_roi_np, 'RGBA')
        else:
            image_pil = Image.fromarray(image_roi_np, 'RGB')
    elif image_roi_np.ndim == 2:
        image_pil = Image.fromarray(image_roi_np).convert('RGB')
    elif image_roi_np.ndim == 3:
        # Assume volumetric data (e.g. shape: (depth, height, width)).
        # Here we take the middle slice for counting.
        d = image_roi_np.shape[0]
        mid_slice = image_roi_np[d // 2]
        if mid_slice.ndim == 2:
            image_pil = Image.fromarray(
                mid_slice.astype(np.uint8), 'L').convert('RGB')
        else:
            image_pil = Image.fromarray(mid_slice.astype(np.uint8), 'RGB')
    else:
        raise ValueError(
            "image_roi must be a 2D or 3D NumPy array or Napari Image Layer.")

    # Process exemplar image if provided.
    exemplar_image_pil = None
    if exemplar_image is not None:
        if isinstance(exemplar_image, np.ndarray):
            exemplar_image_np = exemplar_image
            exemplar_image_pil = Image.fromarray(exemplar_image_np).convert(
                'RGB')  # Convert numpy exemplar to PIL
        elif isinstance(exemplar_image, NapariImage):
            exemplar_image_np = exemplar_image.data
            exemplar_image_pil = Image.fromarray(
                exemplar_image_np, 'RGB')  # Convert numpy exemplar to PIL
        elif isinstance(exemplar_image, Image.Image):
            exemplar_image_pil = exemplar_image  # Already a PIL image.
        else:
            exemplar_image_pil = image_pil

    # For exemplars, we assume exemplar_points (if provided) are already normalized.
    exemplar_boxes_normalized = exemplar_points if exemplar_points is not None else []
    if exemplar_image_pil is None:
        exemplar_image_pil = image_pil

    # Run object counting with ObjectCounter.
    try:  # Added try-except block for better error handling
        detected_boxes_xyxy = object_counter.count_objects(
            image_pil,
            text_prompt=text_prompt,
            exemplar_image=exemplar_image_pil,
            exemplar_boxes=exemplar_boxes_normalized,
            # Adjust threshold as needed.
            confidence_threshold=confidence_threshold,
            # Using text_prompt as keywords for filtering.
            keywords=text_prompt
        )
    except Exception as e:
        # Print detailed error
        print(f"Error during object_counter.count_objects: {e}")
        return [], label_type  # Return empty labels and label_type to avoid further errors

    # Convert detected boxes to the desired output.
    if label_type == 'points':
        points = []
        for box in detected_boxes_xyxy:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            # If the ROI was volumetric (and not a typical 2D color image), add a Z coordinate.
            if (not is_color_2d) and (image_roi_np.ndim == 3) and z_slice is None:
                points.append([center_y, center_x, 0])
            elif z_slice is not None:
                points.append(
                    [z_slice, center_y + offset_y, center_x + offset_x])
            else:
                points.append([center_y, center_x])
        # print(f"Detected {len(points)} for prompt: '{text_prompt}'")
        return np.array(points), 'points'

    elif label_type == 'bboxes':
        bboxes = []
        for box in detected_boxes_xyxy:
            x1, y1, x2, y2 = box
            if (not is_color_2d) and (image_roi_np.ndim == 3):
                # For volumetric images, assume shape is (depth, height, width)
                min_coords = np.array([y1, x1, 0])
                max_coords = np.array([y2, x2, image_roi_np.shape[0]])
            else:
                # For 2D images.
                min_coords = np.array([y1, x1])
                max_coords = np.array([y2, x2])
            bboxes.append((min_coords, max_coords))
        return bboxes, 'bboxes'

    else:
        raise ValueError(f"Unsupported label_type: {label_type}")
