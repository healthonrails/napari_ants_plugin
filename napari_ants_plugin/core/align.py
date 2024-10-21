import ants
import numpy as np
from .utils import create_result_directory

def align_images(fixed_image: np.ndarray, moving_image: np.ndarray, result_dir: str = None, transform_type: str = 'SyNRA', **kwargs) -> np.ndarray:
    """Align two images using ANTs.

    Parameters:
    -----------
    fixed_image : np.ndarray
        The fixed image for alignment.
    moving_image : np.ndarray
        The moving image to be aligned.
    result_dir : str, optional
        Directory to save transformation parameters.
    transform_type : str
        The type of transform to apply (default: 'SyNRA').
    
    Returns:
    --------
    np.ndarray
        The aligned moving image.
    """
    # Create or validate result directory
    result_dir = create_result_directory(result_dir)
    
    # Convert images
    fixed = ants.from_numpy(fixed_image.astype('float32'))
    moving = ants.from_numpy(moving_image.astype('float32'))

    # Perform alignment using ANTs
    result = ants.registration(fixed, moving, type_of_transform=transform_type, outprefix=result_dir, **kwargs)

    # Return the warped moving image as an ndarray
    return result['warpedmovout'].numpy()