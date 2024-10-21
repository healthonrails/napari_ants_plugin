# ANTs Napari Plugin

**ANTs Napari Plugin** is a plugin for [Napari](https://napari.org/) that provides an interface for aligning and transforming volumetric data using [ANTsPy](https://antspy.readthedocs.io/en/latest/), the Python interface for the Advanced Normalization Tools (ANTs).

## Features

- **Image Alignment**: Register two images (fixed and moving) and estimate the transformation parameters.
- **Image Transformation**: Apply transformations to images based on previously estimated parameters.
- **Point Transformation**: Transform coordinate points between images based on the estimated transformation parameters.

## Installation

### Prerequisites

1. **Napari**: Make sure you have Napari installed. You can install it via pip if you haven't already:
   ```bash
   pip install napari
   ```

2.	ANTsPy: You’ll also need ANTsPy, which can be installed with:

	```
	pip install antspyx
	```



### Plugin Installation

Once you have Napari and ANTsPy set up, you can install the ANTs Napari Plugin from your local repository 

Install from Local Repository

If you’re working the plugin locally, you can install it by navigating to the plugin directory and running:

```
pip install .
```

This will install the plugin and its dependencies.

Usage

1.	Launch Napari: Start Napari either from the command line:

``` 
napari
```
Or through Python:
```python
import napari
viewer = napari.Viewer()
```

2.	Open the Plugin:

•	From the Napari menu, navigate to Plugins > ANTs Plugin > Image Alignment or Image Transformation.
•	You can also access the plugin widgets by searching in the “Add Dock Widget” dialog.

### Image Alignment

The Image Alignment widget allows you to register two images (fixed and moving). To use it:

	1.	Load a fixed image and a moving image.
	2.	Choose the transformation type (e.g., SyNRA, Affine).
	3.	Click “Align Images” to estimate the transformation.

Image Transformation

The Image Transformation widget allows you to apply the transformation estimated by the alignment process to the moving image. To use it:

	1.	Select the image and reference.
	2.	Specify the transformation directory.
	3.	Optionally choose whether to invert the transformation and select the interpolation method (e.g., bspline).

### Example Usage (Code)

If you want to use the plugin programmatically:
```python
import napari
import ants

# Start napari viewer
viewer = napari.Viewer()

# Load images
fixed_image = ants.image_read('path/to/fixed_image.nii.gz')
moving_image = ants.image_read('path/to/moving_image.nii.gz')

# Align images
result = ants.registration(fixed_image, moving_image, type_of_transform='SyNRA')

# Display in napari
viewer.add_image(fixed_image.numpy(), name='Fixed Image')
viewer.add_image(result['warpedmovout'].numpy(), name='Transformed Moving Image')

```

### Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request on the GitHub repository.

### License

This plugin is licensed under the MIT License. See the LICENSE file for more details.

### Credits

This plugin was developed by Chen Yang as part of the CPL Lab. It uses the ANTsPy library for image transformations and alignment.

### Contact

For any questions or feedback, please contact Chen Yang at healthonrails@gmail.com.

