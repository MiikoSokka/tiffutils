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
        # Normalize to 0–255 and convert to uint8
        img_min, img_max = img_2d.min(), img_2d.max()
        if img_max > img_min:
            norm_img = ((img_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            norm_img = np.zeros_like(img_2d, dtype=np.uint8)

        edges = cv2.Canny(norm_img, 100, 200)  # Typical thresholds
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
from tiffutils.processing.dtype import convert_dtype  # Assuming you still need this elsewhere

def overlay_arrays(array1: np.ndarray, array2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay array2 on top of array1 with transparency alpha.
    Supports inputs with shape (C, Y, X) or (Z, C, Y, X).
    Assumes array2 is a binary mask (0 or 255), and array1 is in a higher intensity range.
    """
    assert array1.shape == array2.shape, "Input arrays must have the same shape"
    assert 0 <= alpha <= 1, "Alpha must be in [0, 1]"

    if array1.ndim == 3:
        # Single Z slice: CYX
        return _overlay_single(array1, array2, alpha)
    elif array1.ndim == 4:
        # Stack of Z slices: ZCYX
        return np.stack(
            [_overlay_single(array1[z], array2[z], alpha) for z in range(array1.shape[0])],
            axis=0
        )
    else:
        raise ValueError("Input arrays must have 3 (C, Y, X) or 4 (Z, C, Y, X) dimensions")

def _overlay_single(arr1: np.ndarray, arr2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Overlay a single CYX pair of arrays.
    """
    arr1_min = float(arr1.min())
    arr1_max = float(arr1.max())
    arr1_range = arr1_max - arr1_min

    if arr1_range > 1e-8:
        arr1_norm = (arr1.astype(np.float32) - arr1_min) / arr1_range
    else:
        arr1_norm = np.zeros_like(arr1, dtype=np.float32)

    mask = (arr2 > 0).astype(np.float32)

    np.multiply(arr1_norm, (1 - alpha), out=arr1_norm)
    arr1_norm += mask * alpha

    np.multiply(arr1_norm, 255.0, out=arr1_norm)
    return arr1_norm.astype(np.uint8)
