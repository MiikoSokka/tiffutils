# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import cv2

def apply_edges(array):
    """
    Apply edge detection to each 2D binary array (typically segmented masks).

    Supported input array shapes include:
        - (Y, X): A single 2D image.
        - (C, Y, X): A stack of 2D images, e.g., channels.
        - (Z, C, Y, X): A 3D stack with channels.
        - (T, Z, C, Y, X): A time series of 3D channel stacks.

    Each (Y, X) image is assumed to be a binary or label image, where edges denote object boundaries.

    Parameters:
    ----------
    array : np.ndarray
        The input NumPy array of shape 2D–5D containing binary or segmented images.
        Must be convertible to `uint8`, which is required by OpenCV's Canny function.

    Returns:
    -------
    np.ndarray
        A NumPy array of the same shape as the input, where each 2D slice (Y, X) has been replaced
        with its corresponding edge-detected version. The data type is preserved (e.g., `uint8`).

    Raises:
    ------
    ValueError:
        If any image slice passed to edge detection is not 2D.

    Notes:
    -----
    - This function assumes input arrays are binary or low-valued labeled segmentations; applying it 
      to grayscale or natural images may yield undesired edge patterns due to low thresholds (0, 1).
    - The dilation step uses a 1x1 kernel by default; modify this if thicker outlines are needed.
    - Edge slices replace the original input content; no channel insertion is performed.
    """

    def detect_edges_2d(img):
        img = img.astype(np.uint8)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {img.shape}")
        edges = cv2.Canny(img, 0, 1)
        edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
        return edges.astype(img.dtype)

    def recursive_apply(a):
        if a.ndim == 2:
            return detect_edges_2d(a)
        else:
            return np.stack([recursive_apply(sub) for sub in a], axis=0)

    return recursive_apply(array)