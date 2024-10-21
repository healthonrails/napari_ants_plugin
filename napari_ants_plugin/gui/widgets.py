from magicgui import magic_factory
import numpy as np
from napari.layers import Image
from ..core.align import align_images
from ..core.transform import transform_image

@magic_factory(call_button="Align Images")
def image_alignment_widget(fixed_image: "napari.layers.Image", 
                           moving_image: "napari.layers.Image", 
                           transform_type: str = 'SyNRA') -> "napari.layers.Image":
    """Widget for image alignment using ANTs."""
    fixed = fixed_image.data
    moving = moving_image.data
    
    aligned_image = align_images(fixed, moving, transform_type=transform_type)
    
    # Important: Create a new Image layer
    return Image(data=aligned_image)

@magic_factory(call_button="Transform Image")
def image_transformation_widget(image: "napari.layers.Image", 
                               reference: "napari.layers.Image", 
                               transform_dir: str, 
                               invert: bool = False, 
                               interpolation: str = 'bspline') -> "napari.layers.Image":
    """Widget for image transformation using ANTs."""
    img = image.data
    ref = reference.data
    
    transformed_image = transform_image(img, ref, transform_dir, invert=invert, interpolation=interpolation)
    
    # Important: Create a new Image layer
    return Image(data=transformed_image)