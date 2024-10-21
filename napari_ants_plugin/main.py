from napari.plugins import plugin_manager
from .gui.widgets import image_alignment_widget, image_transformation_widget

# Register the widgets with Napari
def napari_experimental_provide_dock_widget():
    return [
        image_alignment_widget,
        image_transformation_widget,
    ]