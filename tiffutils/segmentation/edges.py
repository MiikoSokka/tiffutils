# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import cv2

def apply_edges(array: np.ndarray) -> np.ndarray:
    """
    Apply edge detection to every (Y, X) slice in a ZCYX, CYX, or YX array.

    Parameters
    ----------
    array : np.ndarray
        Input array. Accepted shapes:
            - (Y, X)
            - (C, Y, X)
            - (Z, C, Y, X)

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype as input with edge-detected values.
    """
    input_dtype = array.dtype
    output = np.zeros_like(array)

    def detect_edges_2d(img_2d):
        img_2d = img_2d.astype(np.uint8)
        edges = cv2.Canny(img_2d, 0, 1)
        edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
        return edges.astype(input_dtype)

    if array.ndim == 2:
        output = detect_edges_2d(array)

    elif array.ndim == 3:
        for c in range(array.shape[0]):
            output[c] = detect_edges_2d(array[c])

    elif array.ndim == 4:
        for z in range(array.shape[0]):
            for c in range(array.shape[1]):
                output[z, c] = detect_edges_2d(array[z, c])

    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    return output



import numpy as np
from tiffutils.processing.dtype import convert_dtype

def overlay_arrays(array1: np.ndarray, array2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay array2 on top of array1 with transparency alpha. Assumes array2 is a binary mask (0 or 255),
    and array1 is in a higher intensity range.
    """
    assert array1.shape == array2.shape
    assert 0 <= alpha <= 1

    # Convert to float32
    arr1 = array1.astype(np.float32)
    arr2 = (array2 > 0).astype(np.float32)  # binary mask (0 or 1)

    # Normalize array1 to 0–1 for visualization
    arr1_norm = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-8)

    # Create overlay: add brightness only where array2 > 0
    overlay = arr1_norm * (1 - alpha) + arr2 * alpha

    # Rescale to original dtype
    overlay_scaled = (overlay * 255).astype(np.uint8)
    return overlay_scaled
