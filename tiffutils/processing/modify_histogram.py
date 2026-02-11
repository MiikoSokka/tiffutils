# coding: utf-8
# Author: Miiko Sokka

import numpy as np
from typing import Literal, Tuple
from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)

def histogram_stretch(input_array, intensity_scaling_param=[0, 100]):
    """
    Apply histogram stretching using percentile clipping to a 2D, 3D, 4D, or 5D NumPy array.

    Parameters:
    -----------
    input_array : np.ndarray
        A 2D (Y, X), 3D (Z, Y, X), 4D (Z, C, Y, X), or 5D (T, Z, C, Y, X) NumPy array.
    intensity_scaling_param : list or tuple of two floats
        The lower and upper percentiles used for contrast stretching, e.g., [0, 100].

    Returns:
    --------
    np.ndarray
        The stretched array with the same shape and dtype as the input. The values are clipped
        and scaled to enhance contrast but remain in the original data type's range.
    
    Notes:
    ------
    - For 2D and 3D arrays, stretching is applied globally.
    - For 4D arrays, stretching is applied independently for each channel.
    - For 5D arrays, stretching is applied independently for each timepoint and channel.
    - Stretching does not normalize to [0, 1].
    - For [0, 1] normalization, use tiffutils.processing.dtype(array, dtype=float32)
    """

    t = Timer()

    LOG.debug(
        "start step=histogram_stretch shape=%s dtype=%s intensity_scaling_param=%s",
        getattr(input_array, "shape", None),
        getattr(input_array, "dtype", None),
        intensity_scaling_param,
    )

    p_lower, p_upper = intensity_scaling_param

    if p_lower < 0 or p_upper > 100:
        raise ValueError("\tPercentile values must be between 0 and 100.")

    arr_dtype = input_array.dtype

    if input_array.ndim == 2:
        # print(f'\tHistogram stretching a 2D array of shape {input_array.shape}')
        # 2D: Y, X
        p1 = np.percentile(input_array, p_lower)
        p99 = np.percentile(input_array, p_upper)

        if p99 - p1 < 1e-5:
            LOG.warning("step=histogram_stretch percentile_range_too_small -> returning input")
            return input_array.copy()

        clipped = np.clip(input_array, p1, p99)
        stretched = ((clipped - p1) / (p99 - p1)) * (np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0)

        LOG.info(
            "done step=histogram_stretch ndim=2 intensity_scaling_param=%s time_s=%.3f",
            intensity_scaling_param,
            t.s(),
        )

        return stretched.astype(arr_dtype)

    elif input_array.ndim == 3:
        # print(f'Histogram stretching a 3D array of shape {input_array.shape}')
        # 3D: Z, Y, X
        p1 = np.percentile(input_array, p_lower)
        p99 = np.percentile(input_array, p_upper)

        if p99 - p1 < 1e-5:
            LOG.warning("step=histogram_stretch percentile_range_too_small -> returning input")
            return input_array.copy()

        clipped = np.clip(input_array, p1, p99)
        stretched = ((clipped - p1) / (p99 - p1)) * (np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0)

        LOG.info(
            "done step=histogram_stretch ndim=3 intensity_scaling_param=%s time_s=%.3f",
            intensity_scaling_param,
            t.s(),
        )

        return stretched.astype(arr_dtype)

    elif input_array.ndim == 4:
        # print(f'Histogram stretching a 4D array of shape {input_array.shape}')
        # 4D: Z, C, Y, X
        Z, C, Y, X = input_array.shape
        stretched = np.empty_like(input_array)

        for c in range(C):
            channel = input_array[:, c, :, :]
            p1 = np.percentile(channel, p_lower)
            p99 = np.percentile(channel, p_upper)


            if p99 - p1 < 1e-5:
                LOG.warning(
                    "step=histogram_stretch channel=%d percentile_range_too_small -> pass_through",
                    int(c),
                )
                stretched[:, c, :, :] = channel
            else:
                clipped = np.clip(channel, p1, p99)
                scaled = ((clipped - p1) / (p99 - p1)) * (
                    np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0
                )
                stretched[:, c, :, :] = scaled.astype(arr_dtype)
        
        LOG.info(
            "done step=histogram_stretch ndim=4 channels=%d intensity_scaling_param=%s time_s=%.3f",
            int(C),
            intensity_scaling_param,
            t.s(),
        )
        
        return stretched

    elif input_array.ndim == 5:
        # print(f'Histogram stretching a 5D array of shape {input_array.shape}')
        # 5D: T, Z, C, Y, X
        T, Z, C, Y, X = input_array.shape
        stretched = np.empty_like(input_array)

        for t in range(T):
            for c in range(C):
                subarray = input_array[t, :, c, :, :]
                p1 = np.percentile(subarray, p_lower)
                p99 = np.percentile(subarray, p_upper)

                if p99 - p1 < 1e-5:
                    LOG.warning(
                        "step=histogram_stretch time=%d channel=%d percentile_range_too_small -> pass_through",
                        int(t),
                        int(c),
                    )
                    stretched[t, :, c, :, :] = subarray
                else:
                    clipped = np.clip(subarray, p1, p99)
                    scaled = ((clipped - p1) / (p99 - p1)) * (
                        np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0
                    )
                    stretched[t, :, c, :, :] = scaled.astype(arr_dtype)

        LOG.info(
            "done step=histogram_stretch ndim=5 timepoints=%d channels=%d intensity_scaling_param=%s time_s=%.3f",
            int(T),
            int(C),
            intensity_scaling_param,
            t.s(),
        )
        
        return stretched

    else:
        LOG.warning("Unsupported array with shape %s. Skipping.", input_array.shape)
        return input_array.copy()


# from __future__ import annotations

def _as_float32(x: np.ndarray) -> np.ndarray:
    # stable + fast; enough precision for microscopy preprocessing
    return x.astype(np.float32, copy=False)


def _clip_nonneg(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _restore_dtype(
    x_float: np.ndarray,
    dtype: np.dtype,
    preserve_range: bool = True,
) -> np.ndarray:
    """
    Convert float array back to dtype.
    If preserve_range=True: clip to dtype range (for ints), else leave as float.
    """
    if dtype.kind == "f":
        return x_float.astype(dtype, copy=False)

    if not preserve_range:
        # caller can keep float output if they want; but if they asked for int dtype,
        # this is the safest fallback.
        preserve_range = True

    info = np.iinfo(dtype)
    x = np.clip(x_float, info.min, info.max)
    return x.astype(dtype, copy=False)


def top_hat_background_correct(
    arr: np.ndarray,
    radius_px: int,
    *,
    do_3d: bool = False,
    clip: bool = True,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    White top-hat background correction for punctate/spot-like signals.

    Accepts:
      - ZYX  : (Z, Y, X)
      - ZCYX : (Z, C, Y, X)

    Behavior:
      - By default (do_3d=False), applies 2D white top-hat per Z-slice (and per channel if ZCYX).
      - If do_3d=True and input is ZYX, applies 3D top-hat using a 3D ball footprint.
        (For ZCYX, do_3d=True applies 3D per channel.)

    Parameters
    ----------
    arr : np.ndarray
        Input image, shape ZYX or ZCYX.
    radius_px : int
        Structuring element radius in pixels.
        For spots: pick slightly larger than spot radius.
    do_3d : bool
        Use 3D ball footprint (more expensive). Default False = slice-by-slice 2D.
    clip : bool
        Clip negatives to 0 after correction (usually desired). Default True.
    preserve_dtype : bool
        If True, return same dtype as input (with clipping). If False, return float32.

    Returns
    -------
    np.ndarray
        Corrected array with same shape as input.
    """
    try:
        from skimage.morphology import white_tophat, disk, ball
    except ImportError as e:
        raise ImportError(
            "top_hat_background_correct() requires `skimage.morphology`. "
            "Install them in the environment where you run segmentation."
        ) from e

    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected ZYX (3D) or ZCYX (4D). Got shape {arr.shape}")

    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    in_dtype = arr.dtype
    x = _as_float32(arr)

    if arr.ndim == 3:
        # ZYX
        if do_3d:
            fp = ball(radius_px)
            y = white_tophat(x, footprint=fp)
        else:
            fp = disk(radius_px)
            y = np.empty_like(x)
            for z in range(x.shape[0]):
                y[z] = white_tophat(x[z], footprint=fp)

    else:
        # ZCYX
        zdim, cdim, _, _ = x.shape
        y = np.empty_like(x)

        if do_3d:
            fp3 = ball(radius_px)
            for c in range(cdim):
                y[:, c] = white_tophat(x[:, c], footprint=fp3)
        else:
            fp2 = disk(radius_px)
            for c in range(cdim):
                for z in range(zdim):
                    y[z, c] = white_tophat(x[z, c], footprint=fp2)

    if clip:
        y = _clip_nonneg(y)

    if preserve_dtype:
        return _restore_dtype(y, in_dtype, preserve_range=True)
    return y


def rolling_ball_background_correct(
    arr: np.ndarray,
    radius_px: int,
    *,
    clip: bool = True,
    return_dtype: bool = True,
    preserve_range: bool = True,
    out_max_percentile: float | None = 99.9,  # robust high end
) -> np.ndarray:
    """
Perform rolling-ball background subtraction on a 3D or 4D fluorescence image.

    This function estimates a smooth, low-frequency background using a
    rolling-ball algorithm (morphological opening) and subtracts it from
    the input image. The operation is applied slice-by-slice in Z and,
    if present, independently per channel.

    The primary purpose is **segmentation-oriented preprocessing**, where
    the goal is to suppress diffuse background (e.g. nuclear haze, uneven
    illumination) and enhance true signal, rather than to preserve absolute
    fluorescence intensities.

    Supported input shapes
    ----------------------
    - ZYX  : (Z, Y, X)
    - ZCYX : (Z, C, Y, X)

    Background model
    ----------------
    The rolling-ball algorithm models background as structures that vary
    slowly over space, with a characteristic scale larger than `radius_px`.
    Any features smaller than this scale are preserved; broader features
    are treated as background and subtracted.

    **Important**: If `radius_px` is too small relative to real biological
    structures (e.g. large chromosome territories), the background estimate
    may intrude into true signal and reduce peak intensities
    ("over-subtraction").

    Computation details
    -------------------
    - Computation is performed in float32 for numerical stability.
    - Background subtraction is applied independently to each Z-slice.
      No smoothing or coupling is performed across Z.
    - For multi-channel input (ZCYX), each channel is processed independently.
    - Negative values after subtraction are optionally clipped to zero.

    Output scaling and dtype handling
    ---------------------------------
    By default, the result is rescaled back to the original intensity range
    (e.g. uint16-like) and cast to the input dtype. This is intended to keep
    downstream segmentation code that assumes integer intensity ranges
    behaving consistently.

    Rescaling is performed using a robust maximum:
    - If `out_max_percentile` is not None (default 99.9), the specified
      percentile is used instead of the absolute maximum to avoid outliers
      dominating the scale.
    - If `out_max_percentile` is None, the absolute maximum is used.

    Parameters
    ----------
    arr : np.ndarray
        Input image array with shape (Z, Y, X) or (Z, C, Y, X).
        Integer or floating-point dtypes are supported.
    radius_px : int
        Rolling-ball radius in pixels. Must be larger than the largest
        background structure you want to suppress, and larger than the
        largest true object you want to preserve.
    clip : bool, optional
        If True (default), negative values after background subtraction
        are clipped to zero.
    return_dtype : bool, optional
        If True (default), return the result cast to the input dtype.
        If False, return a float32 array.
    preserve_range : bool, optional
        If True (default) and the input dtype is integer, rescale the
        background-corrected image to approximately match the original
        intensity range before casting.
    out_max_percentile : float or None, optional
        Percentile used to define the robust upper bound for rescaling.
        Set to None to use the absolute maximum instead.

    Returns
    -------
    np.ndarray
        Background-corrected image with the same shape as `arr`.
        Dtype depends on `return_dtype`.

    Notes
    -----
    - This function is designed for **pre-segmentation cleanup**, not
      quantitative intensity measurements.
    - For punctate signals (spots, foci), a white top-hat or DoG filter
      may be more appropriate.
    - For severe field-wide illumination gradients, consider flat-field
      or shading correction prior to rolling-ball subtraction.

    Raises
    ------
    ValueError
        If `arr` does not have 3 or 4 dimensions, or if `radius_px <= 0`.
    """
    try:
        from skimage.restoration import rolling_ball
        from skimage.morphology import disk
    except ImportError as e:
        raise ImportError(
            "rolling_ball_background_correct() requires `skimage.restoration`. "
            "Install them in the environment where you run segmentation."
        ) from e

    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected ZYX or ZCYX. Got {arr.shape}")
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    in_dtype = arr.dtype
    x = arr.astype(np.float32, copy=False)

    # footprint for your skimage version
    k = disk(radius_px)

    def rb2d(img2d: np.ndarray) -> np.ndarray:
        bg = rolling_ball(img2d, radius=radius_px, kernel=k)
        y = img2d - bg
        return np.maximum(y, 0) if clip else y

    if x.ndim == 3:
        y = np.empty_like(x)
        for z in range(x.shape[0]):
            y[z] = rb2d(x[z])
    else:
        zdim, cdim, _, _ = x.shape
        y = np.empty_like(x)
        for c in range(cdim):
            for z in range(zdim):
                y[z, c] = rb2d(x[z, c])

    if not return_dtype:
        return y  # float32

    # Cast back, optionally preserving original "uint16-like" range
    if in_dtype.kind in ("u", "i") and preserve_range:
        # choose a robust scaling target to avoid outliers exploding the scale
        if out_max_percentile is None:
            src_max = float(np.max(y))
            dst_max = float(np.max(x))
        else:
            src_max = float(np.percentile(y, out_max_percentile))
            dst_max = float(np.percentile(x, out_max_percentile))

        if src_max > 0:
            y = y * (dst_max / src_max)

        info = np.iinfo(in_dtype)
        y = np.clip(y, info.min, info.max).astype(in_dtype, copy=False)
        return y

    # float input dtype case
    return y.astype(in_dtype, copy=False)