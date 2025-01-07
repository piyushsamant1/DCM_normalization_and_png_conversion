
import tensorflow as tf
import numpy as np
from skimage import morphology

def load_segmentation_model(model_path):
    """Loads the pre-trained segmentation model."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_loss": lambda y_true, y_pred: 1.0,
            "iou": lambda y_true, y_pred: 1.0,
            "dice_coef": lambda y_true, y_pred: 1.0
        },
        compile=False
    )

def predict_mask(segmentation_model, patch):
    """Runs the segmentation model on the patch and returns the predicted mask."""
    patch_array = np.stack((patch,) * 3, axis=-1)  # Convert to 3 channels
    patch_array = patch_array / 255.0  # Normalize
    patch_array = np.expand_dims(patch_array, axis=0)  # Add batch dimension
    prediction = segmentation_model.predict(patch_array)
    predicted_mask = prediction[0, :, :, 0]  # Extract the mask
    return (predicted_mask * 255).astype(np.uint8)

def post_process_mask(mask):
    """Applies common post-processing steps like morphological operations."""
    processed_mask = morphology.binary_closing(mask > 128, morphology.disk(3))  # Example: closing
    return (processed_mask * 255).astype(np.uint8)
