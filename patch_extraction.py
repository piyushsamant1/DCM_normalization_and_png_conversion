# from PIL import Image
# import numpy as np
# from skimage.morphology import binary_closing, disk
# def extract_patch(image, center_point, patch_size=128):
#     """
#     Extracts a patch from the image centered around the specified point.
#     If the patch exceeds image boundaries, it is adjusted accordingly.
#     :param image: The full image array (2D or 3D).
#     :param center_point: Tuple (cx, cy) for the center of the patch.
#     :param patch_size: Size of the patch (128x128 or 64x64).
#     :return: The extracted patch.
#     """
#     cx, cy = center_point
#     half_size = patch_size // 2

#     # Calculate the bounding box for the patch
#     start_x = max(0, cx - half_size)
#     start_y = max(0, cy - half_size)
#     end_x = min(image.shape[1], cx + half_size)
#     end_y = min(image.shape[0], cy + half_size)

#     # Adjust if the patch is near the image boundary
#     if end_x - start_x != patch_size:
#         start_x = end_x - patch_size
#     if end_y - start_y != patch_size:
#         start_y = end_y - patch_size
    
#     patch = image[start_y:end_y, start_x:end_x]
#     return patch

# def sliding_window(image, window_size=128, stride=64):
#     """
#     Extracts patches from the image using a sliding window.
#     :param image: Input image array (assumed to be 2D grayscale).
#     :param window_size: Size of each patch.
#     :param stride: Overlap between patches.
#     :return: A list of patches.
#     """
#     patches = []
#     h, w = image.shape
#     for y in range(0, h - window_size + 1, stride):
#         for x in range(0, w - window_size + 1, stride):
#             patch = image[y:y + window_size, x:x + window_size]
#             patches.append((patch, (x, y)))  # Store the patch and its top-left coordinates
#     return patches

# def extract_nodule_patch(image_patch, mask_patch):
#     """
#     Extracts the nodule-only patch by applying the mask to the image patch.
#     :param image_patch: The 128x128 image patch (either grayscale or multi-channel).
#     :param mask_patch: The 128x128 predicted mask from the segmentation model (binary or grayscale).
#     :return: Nodule-only image patch where the background is set to zero.
#     """
#     # Ensure the mask is binary (0 or 1)
#     binary_mask = mask_patch > 10  # Threshold the mask
   
#     # If the image patch is grayscale (2D array), apply the mask directly
#     if len(image_patch.shape) == 2:
#         nodule_patch = np.multiply(image_patch, binary_mask)
    
#     # If the image patch is RGB (3D array), apply the mask to each channel
#     elif len(image_patch.shape) == 3 and image_patch.shape[2] == 3:
#         nodule_patch = np.zeros_like(image_patch)
#         for i in range(3):
#             nodule_patch[:, :, i] = image_patch[:, :, i] * binary_mask
    
#     else:
#         raise ValueError("Unexpected image patch shape: should be either 2D (grayscale) or 3D (RGB).")
    
#     return nodule_patch

# def process_and_save_patch(image, full_mask, center_point, model, save_prefix, patch_size=128):
#     """
#     Extracts a patch, applies the segmentation model, and saves the patch and the predicted mask.
#     Morphological operations are applied to the predicted mask to fill gaps.
    
#     :param image: The full image array (2D or 3D).
#     :param full_mask: The full predicted mask for the image.
#     :param center_point: Tuple (cx, cy) for the center of the patch.
#     :param patch_size: The size of the patch to be extracted (128 or 64).
#     :param model: The segmentation model.
#     :param save_prefix: The prefix for saving files.
#     :return: The extracted patch, the mask from the full image, and the predicted mask after applying morphological operations.
#     """
#     # Step 1: Extract the patch and mask from the full image and full mask
#     patch = extract_patch(image, center_point, patch_size=patch_size)
#     mask_patch = extract_patch(full_mask, center_point, patch_size=patch_size)
        
#     # Step 2: Apply the segmentation model on the patch
#     predicted_mask = predict_mask(model, patch)
    
#     # Step 3: Apply morphological operations (closing) to fill gaps in the predicted mask
#     # Binary closing with a disk-shaped structuring element (you can adjust the size)
#     closed_mask = binary_closing(predicted_mask > 128, disk(3))  # Threshold to binary and apply closing
    
#     # Step 4: Save the patch and the predicted mask after morphological closing
#     patch_image = Image.fromarray(patch)
#     mask_image = Image.fromarray((closed_mask * 255).astype(np.uint8))  # Convert binary mask back to 8-bit
    
#     patch_image.save(f"{save_prefix}_patch_{patch_size}x{patch_size}.png")
#     mask_image.save(f"{save_prefix}_mask_{patch_size}x{patch_size}_morph.png")
    
#     print(f"Saved patch and mask for size {patch_size}x{patch_size} after morphological closing.")
    
#     return patch, mask_patch, closed_mask



from PIL import Image
import numpy as np
from skimage.morphology import binary_closing, disk
from nodule_segmentation import predict_mask  # Import only the necessary function to avoid circular dependency

def extract_patch(image, center_point, patch_size=128):
    """Extracts a patch from the image centered around the specified point."""
    cx, cy = center_point
    half_size = patch_size // 2
    start_x = max(0, cx - half_size)
    start_y = max(0, cy - half_size)
    end_x = min(image.shape[1], cx + half_size)
    end_y = min(image.shape[0], cy + half_size)
    if end_x - start_x != patch_size:
        start_x = end_x - patch_size
    if end_y - start_y != patch_size:
        start_y = end_y - patch_size
    patch = image[start_y:end_y, start_x:end_x]
    return patch

def sliding_window(image, window_size=128, stride=64):
    """Extracts patches from the image using a sliding window."""
    patches = []
    h, w = image.shape
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y + window_size, x:x + window_size]
            patches.append((patch, (x, y)))  # Store the patch and its top-left coordinates
    return patches

def extract_nodule_patch(image_patch, mask_patch):
    """Extracts the nodule-only patch by applying the mask to the image patch."""
    binary_mask = mask_patch > 10
    if len(image_patch.shape) == 2:
        nodule_patch = np.multiply(image_patch, binary_mask)
    elif len(image_patch.shape) == 3 and image_patch.shape[2] == 3:
        nodule_patch = np.zeros_like(image_patch)
        for i in range(3):
            nodule_patch[:, :, i] = image_patch[:, :, i] * binary_mask
    else:
        raise ValueError("Unexpected image patch shape: should be either 2D (grayscale) or 3D (RGB).")
    return nodule_patch

def process_and_save_patch(image, full_mask, center_point, model, save_prefix, patch_size=128):
    """Extracts a patch, applies the segmentation model, and saves the patch and the predicted mask."""
    patch = extract_patch(image, center_point, patch_size=patch_size)
    mask_patch = extract_patch(full_mask, center_point, patch_size=patch_size)
    predicted_mask = predict_mask(model, patch)
    closed_mask = binary_closing(predicted_mask > 128, disk(3))
    patch_image = Image.fromarray(patch)
    mask_image = Image.fromarray((closed_mask * 255).astype(np.uint8))
    patch_image.save(f"{save_prefix}_patch_{patch_size}x{patch_size}.png")
    mask_image.save(f"{save_prefix}_mask_{patch_size}x{patch_size}_morph.png")
    return patch, mask_patch, closed_mask

