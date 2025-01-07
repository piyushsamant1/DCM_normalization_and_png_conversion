import pydicom
from PIL import Image
import numpy as np

def read_dicom(file_path):
    """Reads a DICOM file and checks if the modality is 'CT'."""
    try:
        ds = pydicom.dcmread(file_path)
        if ds.Modality != "CT":
            raise ValueError("DICOM file is not a CT scan. Exiting.")
        return ds
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        exit(1)

def normalize_and_convert(ds, window_center=-600, window_width=1800, use_16bit_depth=False):
    """
    Normalizes and converts the DICOM image to 8-bit (or 16-bit if specified) PNG.
    """
    pixel_array = ds.pixel_array

    # Adjust for rescale slope and intercept
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept

    # Apply windowing
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    pixel_array = np.clip(pixel_array, min_value, max_value)

    # Normalize to 8-bit
    pixel_array = ((pixel_array - min_value) / (max_value - min_value)) * 255
    pixel_array = pixel_array.astype(np.uint8)

    # Create PIL image
    image = Image.fromarray(pixel_array)
    if use_16bit_depth:
        image = image.convert("I")

    return image

def save_image(image, output_path):
    """Saves the image to a specified output path."""
    image.save(output_path)
    print(f"Image saved at {output_path}")
