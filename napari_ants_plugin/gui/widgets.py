from magicgui import magic_factory
from napari.layers import Image
from ..core.align import align_images
from ..core.transform import transform_image
from napari.types import LayerDataTuple
from napari import Viewer
from skimage.io import imread
import os

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