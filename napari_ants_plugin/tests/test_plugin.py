import numpy as np
from napari_ants_plugin.core.align import align_images
from napari_ants_plugin.core.transform import transform_image

def test_align_images():
    fixed_image = np.random.rand(100, 100)
    moving_image = np.random.rand(100, 100)
    
    result = align_images(fixed_image, moving_image)
    
    assert result.shape == fixed_image.shape

def test_transform_image():
    image = np.random.rand(100, 100)
    reference = np.random.rand(100, 100)
    transform_dir = '/path/to/transforms'
    
    result = transform_image(image, reference, transform_dir)
    
    assert result.shape == image.shape