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


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from typing import Union
import tiffutils as tiffu
from joblib import Parallel, delayed

def montage_segmentation_edges(array: np.ndarray, alpha: float = 0.5, save_as: Union[str, None] = None):
    """
    This is a function to create an overlay montage of raw images and segmentation masks made to edges.

    The raw images need to be in channel 1 and the edge images (can be binary or 8-bit,
    but need to have two values [0, >=1]) in channel 2. Cannot use raw segmentations, but they need
    to be modifed to edges by using tiffu.apply_edges() function.

    For multiple raw images, they all need to be in T or Z layer. For example, if you have
    24 raw images of targets, your stack arrangement need to be (24, 2, Y, X) or (24, 1, 2, Y, X).

    Create a 4x6 montage of composite images from an input NumPy array with shape:
    - (T, Z, C, Y, X) where Z = 1 and C = 2
    - (Z, C, Y, X) where Z = 24 and C = 2

    The first channel is mapped with the 'inferno' colormap (a modern replacement for "fire"),
    and the second channel is a binary mask mapped with the 'gray' colormap, shown with controllable transparency.

    Parameters:
    - array: Input numpy array
    - alpha: Transparency for where the second channel equals 1 (0 = invisible, 1 = fully opaque)
    - save_as: If provided, the output montage is saved to this filepath
    """

    # Handle shape
    if array.ndim == 5:
        T, Z, C, Y, X = array.shape
        assert Z == 1 and C == 2, "Expected shape (T,1,2,Y,X)"
        images = array[:, 0, :, :, :]  # shape (T, 2, Y, X)
    elif array.ndim == 4:
        Z, C, Y, X = array.shape
        assert C == 2, "Expected shape (Z,2,Y,X)"
        images = array  # shape (Z, 2, Y, X)
    else:
        raise ValueError("Array must be 4D (Z,2,Y,X) or 5D (T,1,2,Y,X)")

    num_images = images.shape[0]
    assert num_images <= 24, "Maximum 24 images for 4x6 montage"

    # Colormaps
    cmap1 = colormaps["inferno"]
    cmap2 = colormaps["gray"]

    composites = []
    for i in range(num_images):
        ch1 = images[i, 0]
        ch2 = images[i, 1]

        # Normalize channel 1
        norm1 = Normalize()(ch1)
        rgb1 = cmap1(norm1)[..., :3]  # (Y, X, 3)

        # Binary mask for channel 2 (works for binary [0, 1] or values >= 1)
        mask = (ch2 >= 1)

        # Create gray overlay
        gray_overlay = np.ones((*ch2.shape, 3))  # solid white RGB

        # Create alpha mask: 0 where ch2 == 0, else `alpha`
        alpha_mask = np.where(mask, alpha, 0.0)[..., None]  # (Y, X, 1)

        # Composite: overlay gray on top of fire using alpha mask
        composite = gray_overlay * alpha_mask + rgb1 * (1 - alpha_mask)
        composites.append(composite)

    # Create montage
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < len(composites):
            ax.imshow(composites[i])

    # Set consistent spacing (3 pt)
    spacing_in = 3 / 72  # convert points to inches
    fig.subplots_adjust(
        wspace=spacing_in / (18 / 6),  # adjust for 6 columns in 18-inch width
        hspace=spacing_in / (12 / 4)   # adjust for 4 rows in 12-inch height
    )


    plt.tight_layout()
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as, dpi=300)
    plt.close(fig)
