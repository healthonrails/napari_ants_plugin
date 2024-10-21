import os
import ants

def create_result_directory(result_dir: str) -> str:
    """Create or validate a result directory for saving transformations."""
    if not result_dir:
        result_dir = os.path.join(os.getcwd(), 'ANTs_results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def compose_transforms(transform_dir: str, invert: bool = False):
    """Compose a set of ANTs transforms from a directory.

    Parameters:
    -----------
    transform_dir : str
        Directory with ANTs transform files.
    invert : bool, optional
        Whether to invert the transformation.

    Returns:
    --------
    ants.ANTsTransform
        A composite transformation object.
    """
    transforms = []
    if not invert:
        if '1Warp.nii.gz' in os.listdir(transform_dir):
            SyN_file = os.path.join(transform_dir, '1Warp.nii.gz')
            field = ants.image_read(SyN_file)
            transform = ants.transform_from_displacement_field(field)
            transforms.append(transform)
        if '0GenericAffine.mat' in os.listdir(transform_dir):
            affine_file = os.path.join(transform_dir, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file))
    else:
        if '0GenericAffine.mat' in os.listdir(transform_dir):
            affine_file = os.path.join(transform_dir, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file).invert())
        if '1InverseWarp.nii.gz' in os.listdir(transform_dir):
            inv_file = os.path.join(transform_dir, '1InverseWarp.nii.gz')
            field = ants.image_read(inv_file)
            transform = ants.transform_from_displacement_field(field)
            transforms.append(transform)
    
    return ants.compose_ants_transforms(transforms)