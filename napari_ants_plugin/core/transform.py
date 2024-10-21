import ants
import numpy as np
from .utils import compose_transforms


def transform_image(image: np.ndarray, reference: np.ndarray, transform_dir: str, invert: bool = False, interpolation: str = 'bspline') -> np.ndarray:
    """Transform an image using a precomputed transformation.

    Parameters:
    -----------
    image : np.ndarray
        The image to transform.
    reference : np.ndarray
        The reference image for the transform.
    transform_dir : str
        Directory with transformation parameters.
    invert : bool, optional
        Whether to invert the transformation.
    interpolation : str, optional
        Interpolation method (default: 'bspline').

    Returns:
    --------
    np.ndarray
        The transformed image.
    """
    # Load images using ANTs
    im = ants.from_numpy(image.astype('float32'))
    ref = ants.from_numpy(reference.astype('float32'))

    # Compose and apply transforms
    composite_transform = compose_transforms(transform_dir, invert=invert)
    result = composite_transform.apply_to_image(
        im, ref, interpolation=interpolation)

    return result.numpy()
