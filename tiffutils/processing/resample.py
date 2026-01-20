# coding: utf-8
# Author: Miiko Sokka

import numpy as np
from scipy.ndimage import zoom

from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)

def resample_z_to_match_xy(
    arr: np.ndarray,
    xy_pixel_size_nm: float = 94.917,
    z_pixel_size_nm: float = 500.039,
    order: int = 1,
    ) -> np.ndarray:
    """
    Resample the Z axis of a 3D (Z, Y, X) or 4D (Z, C, Y, X) array so that
    the physical spacing in Z matches the XY pixel size, by interpolation.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (Z, Y, X) or (Z, C, Y, X).
    xy_pixel_size_nm : float
        Pixel size in XY (nm per pixel).
    z_pixel_size_nm : float
        Pixel size in Z (nm per slice).
    order : int
        Spline interpolation order for scipy.ndimage.zoom.
        0 = nearest, 1 = linear (default), 3 = cubic, etc.

    Returns
    -------
    np.ndarray
        Array with the same number of XY pixels (and channels, if present),
        but with Z interpolated to have approximately isotropic voxels.
    """
    t = Timer()

    LOG.debug(
        "start step=resample_z_to_match_xy xy_pixel_size_nm=%s z_pixel_size_nm=%s order=%d",
        xy_pixel_size_nm,
        z_pixel_size_nm,
        order,
        )


    if arr.ndim not in (3, 4):
        raise ValueError(
            f"Expected a 3D (Z, Y, X) or 4D (Z, C, Y, X) array, got shape {arr.shape}"
        )

    # Original Z size
    z_orig = arr.shape[0]

    # Factor by which the number of Z slices should change to match XY spacing
    # Example: 500.039 / 94.917 ≈ 5.27 → ~5.27x more Z slices
    z_scale = z_pixel_size_nm / xy_pixel_size_nm
    z_new = int(round(z_orig * z_scale))


    if z_new <= 0:
        raise ValueError(
            f"Computed non-positive target Z size ({z_new}). "
            f"Check xy_pixel_size_nm={xy_pixel_size_nm} and z_pixel_size_nm={z_pixel_size_nm}."
        )

    # Build zoom factors: only change Z, keep others the same
    zoom_factors = [z_new / z_orig]
    if arr.ndim == 3:
        zoom_factors.extend([1.0, 1.0])      # (Z, Y, X)
    else:  # 4D: (Z, C, Y, X)
        zoom_factors.extend([1.0, 1.0, 1.0])

    orig_dtype = arr.dtype

    # Perform interpolation in float for safety
    arr_float = arr.astype(np.float32, copy=False)
    resampled = zoom(arr_float, zoom=zoom_factors, order=order)

    # Clip and cast back to original dtype if it is integer
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        resampled = np.clip(resampled, info.min, info.max).astype(orig_dtype)
    else:
        resampled = resampled.astype(orig_dtype, copy=False)

    LOG.info(
        "done step=resample_z_to_match_xy shape_in=%s shape_out=%s time_s=%.3f",
        arr.shape,
        resampled.shape,
        t.s(),
    )

    return resampled