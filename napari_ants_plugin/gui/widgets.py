from magicgui import magic_factory
from napari.layers import Image
from ..core.align import align_images
from ..core.transform import transform_image
from napari.types import LayerDataTuple
from napari import Viewer
from skimage.io import imread
import os
from ..core.utils import log_score

@magic_factory(call_button="Align Images",
               fixed_image_path={"widget_type": "FileEdit", "label": "Fixed Image File"},
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
        raise FileNotFoundError(f"Fixed image file not found: {fixed_image_path}")
    if not os.path.exists(moving_image_path):
        raise FileNotFoundError(f"Moving image file not found: {moving_image_path}")

    # Load the images from the file paths
    fixed_image_data = imread(fixed_image_path)
    moving_image_data = imread(moving_image_path)

    # Add images to the viewer only if they are not already added
    if "Fixed Image" not in viewer.layers:
        viewer.add_image(fixed_image_data, name="Fixed Image")
    if "Moving Image" not in viewer.layers:
        viewer.add_image(moving_image_data, name="Moving Image")

    # Perform alignment
    aligned_image = align_images(fixed_image_data, moving_image_data, transform_type=transform_type)

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
        raise FileNotFoundError(f"Reference image file not found: {reference_image_path}")

    # Load the images from the file paths
    image_data = imread(image_path)
    reference_image_data = imread(reference_image_path)

    # Add images to the viewer only if they are not already added
    if "Image" not in viewer.layers:
        viewer.add_image(image_data, name="Image")
    if "Reference Image" not in viewer.layers:
        viewer.add_image(reference_image_data, name="Reference Image")

    # Perform the transformation
    transformed_image = transform_image(image_data, reference_image_data, transform_dir, invert=invert, interpolation=interpolation)

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
            raise ValueError("No 'Aligned Image' found in the viewer. Please provide a path to load one.")
        
        if not os.path.exists(aligned_image_path):
            raise FileNotFoundError(f"Aligned image file not found: {aligned_image_path}")
        
        # Load the aligned image from the specified path into the viewer
        aligned_image_data = imread(aligned_image_path)
        
        # Add the aligned image to the viewer as a new layer
        aligned_image_layer = viewer.add_image(aligned_image_data, name="Aligned Image")

    # Ensure the score is within the valid range
    if not (0 <= score <= 10):
        raise ValueError("Score must be between 0 and 10.")

    # Log the score and comments
    log_score(aligned_image_layer.name, score, comments)