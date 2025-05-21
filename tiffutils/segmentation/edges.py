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
    Overlay array2 on top of array1 with a given transparency.

    Parameters:
    -----------
    array1 : np.ndarray
        The base array. Must be of shape (C, Y, X) or (Z, C, Y, X).
    array2 : np.ndarray
        The overlay array. Must be the same shape as array1. Will be converted to array1's dtype.
    alpha : float
        Transparency level for array2. Must be between 0 (fully transparent) and 1 (fully opaque).

    Returns:
    --------
    np.ndarray
        The resulting overlay array with the same shape and dtype as array1.

    Raises:
    -------
    AssertionError
        If array1 and array2 shapes are not equal.
    """
    assert array1.shape == array2.shape, "Input arrays must have the same shape."
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."

    array2_converted = convert_dtype(array2, str(array1.dtype))

    # Blend using float32 to avoid intermediate precision issues
    result = (1 - alpha) * array1.astype(np.float32) + alpha * array2_converted.astype(np.float32)

    # Clip and cast back to original dtype if it's integer
    if np.issubdtype(array1.dtype, np.integer):
        info = np.iinfo(array1.dtype)
        result = np.clip(result, info.min, info.max)

    return result.astype(array1.dtype)
