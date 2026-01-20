# coding: utf-8
# Author: Miiko Sokka

from __future__ import annotations

import numpy as np

from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)


def segmentMetaphaseChromosomes(
    array_3D: np.ndarray,
    f3_param: list[list[float]] = [[1, 0.01]],
    minArea: int = 4,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Segment chromosome-like filaments from a 3D fluorescence image.

    Parameters
    ----------
    array_3D : np.ndarray
        A 3D NumPy array of shape (Z, Y, X), expected to be histogram-stretched.
    f3_param : list of list, optional
        Parameters for the `filament_3d_wrapper` function, controlling filament detection.
        Default is ``[[1, 0.01]]``. This works well for metaphase chromosome spreads.
    minArea : int, optional
        Minimum size (in voxels) for an object to be retained in the post-processing step.
        Default is 4.
    verbose : bool, optional
        If True, emit INFO-level progress logs.

    Returns
    -------
    np.ndarray
        A 3D array of the same shape as `array_3D`, dtype uint16.
        Values are 0 for background and 65535 for segmented objects.
    """
    t = Timer()

    if array_3D.ndim != 3:
        raise ValueError(f"array_3D must be 3D (Z,Y,X); got shape={array_3D.shape}")

    if verbose:
        LOG.info(
            "segmentMetaphaseChromosomes start shape=%s dtype=%s f3_param=%s minArea=%s",
            array_3D.shape,
            array_3D.dtype,
            f3_param,
            minArea,
        )

    try:
        from aicssegmentation.core.vessel import filament_3d_wrapper
        from skimage.morphology import remove_small_objects
    except ImportError as e:
        raise ImportError(
            "segmentMetaphaseChromosomes() requires `aicssegmentation` and `scikit-image`. "
            "Install them in the environment where you run segmentation."
        ) from e

    # Segmentation
    bw = filament_3d_wrapper(array_3D, f3_param)

    # Post processing
    seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
    seg_uint16 = (seg * 65535).astype(np.uint16)

    if verbose:
        fg = int(np.count_nonzero(seg_uint16))
        LOG.info(
            "segmentMetaphaseChromosomes done fg_voxels=%s time_s=%.3f",
            fg,
            t.s(),
        )

    return seg_uint16


def segmentChromosomeTerritories(
    array_zyx: np.ndarray,
    intensity_scaling_param: list[float] | tuple[float, ...] = [0.8, 8.5, 300, 650],
    gaussian_smoothing_sigma: float = 0.7,
    s2_param_bright=None,
    s2_param_dark=None,
    threshold: str = "tri",
    min_area: int = 700,
    apply_intensity_normalization: bool = True,
    relative_max_intensity_threshold: float = 0.35,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Segment chromosome territories from a 3D fluorescence image.

    Uses AICS-segmentation Masked-Object (MO) thresholding and an object-level
    relative max-intensity filter to suppress faint "ghost" segmentations.

    Parameters
    ----------
    array_zyx : np.ndarray
        3D input image with shape (Z, Y, X).
    intensity_scaling_param : tuple or list, optional
        Parameters passed to `intensity_normalization` when
        `apply_intensity_normalization` is True.
    gaussian_smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing in voxels.
    s2_param_bright, s2_param_dark
        Kept for API compatibility (currently not used).
    threshold : str or float, optional
        Global thresholding method passed to `MO`.
    min_area : int, optional
        Minimum object size (in voxels) for MO thresholding.
    apply_intensity_normalization : bool, optional
        If True, apply `intensity_normalization` before smoothing.
    relative_max_intensity_threshold : float, optional
        Keep only components with max intensity >= this fraction of the brightest
        component's max intensity.
    verbose : bool, optional
        If True, emit INFO-level progress logs.

    Returns
    -------
    np.ndarray
        A 3D array of the same shape and dtype as `array_zyx`.
        Background voxels are 0; foreground voxels are set to the max value of
        the input dtype.
    """
    t = Timer()

    try:
        from aicssegmentation.core.MO_threshold import MO
        from aicssegmentation.core.pre_processing_utils import (
            intensity_normalization,
            image_smoothing_gaussian_3d,
        )
        from skimage.morphology import remove_small_objects
        from skimage.measure import label, regionprops
    except ImportError as e:
        raise ImportError(
            "segmentChromosomeTerritories() requires `aicssegmentation` and `scikit-image`. "
            "Install them in the environment where you run it."
        ) from e

    if array_zyx.ndim != 3:
        raise ValueError(f"array_zyx must be 3D (Z,Y,X); got shape={array_zyx.shape}")

    if not (0.0 <= float(relative_max_intensity_threshold) <= 1.0):
        raise ValueError("relative_max_intensity_threshold must be between 0 and 1 (inclusive).")

    # Defaults for dot filters (kept for API compatibility)
    if s2_param_bright is None:
        s2_param_bright = [[2, 0.025]]
    if s2_param_dark is None:
        s2_param_dark = [[2, 0.025], [1, 0.025]]

    orig_dtype = array_zyx.dtype

    if verbose:
        LOG.info(
            "segmentChromosomeTerritories start shape=%s dtype=%s threshold=%s min_area=%s sigma=%s apply_norm=%s relmax_thr=%s",
            array_zyx.shape,
            orig_dtype,
            threshold,
            min_area,
            gaussian_smoothing_sigma,
            apply_intensity_normalization,
            relative_max_intensity_threshold,
        )

    img = array_zyx.astype(np.float32, copy=False)

    if apply_intensity_normalization:
        img = intensity_normalization(img, scaling_param=intensity_scaling_param)

    img_smooth = image_smoothing_gaussian_3d(img, sigma=gaussian_smoothing_sigma)

    bw_mo, _ = MO(
        img_smooth,
        global_thresh_method=threshold,
        object_minArea=min_area,
        return_object=True,
    )

    bw_merge = bw_mo

    # Remove small objects (binary)
    bw_merge = remove_small_objects(bw_merge > 0, min_size=1000, connectivity=0)

    # Object-level intensity filter using relative max intensity
    lab = label(bw_merge)
    if lab.max() > 0:
        props = regionprops(lab, intensity_image=img_smooth)
        max_ints = np.array([p.max_intensity for p in props], dtype=np.float32)
        best = float(max_ints.max())
        if best > 0:
            rel = max_ints / (best + 1e-8)
            keep_labels = [
                props[i].label
                for i in range(len(props))
                if rel[i] >= relative_max_intensity_threshold
            ]
            bw_merge = np.isin(lab, keep_labels)

    # Convert boolean mask to same dtype as input
    if np.issubdtype(orig_dtype, np.integer):
        fg_val = np.iinfo(orig_dtype).max
    elif np.issubdtype(orig_dtype, np.floating):
        fg_val = np.array(1.0, dtype=orig_dtype)
    else:
        fg_val = 1

    out = np.zeros_like(array_zyx, dtype=orig_dtype)
    out[bw_merge] = fg_val

    if verbose:
        n_fg = int(np.count_nonzero(out))
        LOG.info("segmentChromosomeTerritories done fg_voxels=%s time_s=%.3f", n_fg, t.s())

    return out
