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
    intensity_scaling_param: list[float] | tuple[float, ...] = (0.8, 8.5, 300, 650),
    gaussian_smoothing_sigma: float = 0.7,
    s2_param_bright=None,
    s2_param_dark=None,
    threshold: str | float = "tri",
    min_area: int = 700,
    apply_intensity_normalization: bool = True,
    relative_max_intensity_threshold: float = 0.35,
    *,
    min_region_voxels: int = 0,
    relmax_filter_stage: str = "post",   # "pre", "post", or "both"
    relabel_sequential: bool = True,
    binary_min_size: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """Segment chromosome territories from a 3D fluorescence image, returning integer labels (int32).

    Background = 0; territories = 1..N.

    Notes
    -----
    - Uses MO thresholding to create a foreground mask.
    - Uses SciPy Gaussian filter for smoothing.
    - Connected components are used directly as territory labels.
    """
    import time
    import logging
    import numpy as np

    class _Timer:
        def __init__(self):
            self._t0 = time.time()

        def s(self) -> float:
            return time.time() - self._t0

    t = _Timer()
    LOG_ = globals().get("LOG", logging.getLogger(__name__))

    try:
        from aicssegmentation.core.MO_threshold import MO
        from aicssegmentation.core.pre_processing_utils import intensity_normalization
        from scipy import ndimage as ndi
        from skimage.measure import label, regionprops
        from skimage.morphology import remove_small_objects
        from skimage.segmentation import relabel_sequential as _relabel_sequential
    except ImportError as e:
        raise ImportError(
            "segmentChromosomeTerritories() requires `aicssegmentation`, `scikit-image`, and `scipy`."
        ) from e

    if array_zyx.ndim != 3:
        raise ValueError(f"array_zyx must be 3D (Z,Y,X); got shape={array_zyx.shape}")

    if not (0.0 <= float(relative_max_intensity_threshold) <= 1.0):
        raise ValueError("relative_max_intensity_threshold must be between 0 and 1 (inclusive).")

    if relmax_filter_stage not in {"pre", "post", "both"}:
        raise ValueError('relmax_filter_stage must be one of {"pre","post","both"}.')

    # Defaults for dot filters (kept for API compatibility)
    if s2_param_bright is None:
        s2_param_bright = [[2, 0.025]]
    if s2_param_dark is None:
        s2_param_dark = [[2, 0.025], [1, 0.025]]

    if verbose:
        LOG_.info(
            "segmentChromosomeTerritories start shape=%s dtype=%s threshold=%s min_area=%s "
            "sigma=%s apply_norm=%s relmax_filter_stage=%s binary_min_size=%s",
            array_zyx.shape,
            array_zyx.dtype,
            threshold,
            min_area,
            gaussian_smoothing_sigma,
            apply_intensity_normalization,
            relmax_filter_stage,
            binary_min_size,
        )

    # float32 working image
    img = array_zyx.astype(np.float32, copy=False)

    if apply_intensity_normalization:
        img = intensity_normalization(img, scaling_param=list(intensity_scaling_param))

    # ---- smoothing ----
    img_smooth = ndi.gaussian_filter(img, sigma=float(gaussian_smoothing_sigma))

    # ---- MO thresholding ----
    bw_mo, _ = MO(
        img_smooth,
        global_thresh_method=threshold,
        object_minArea=min_area,
        return_object=True,
    )

    # ---- Binary cleanup ----
    bw_merge = (bw_mo > 0)
    bw_merge = remove_small_objects(
        bw_merge,
        min_size=int(binary_min_size),
    )

    def _apply_relmax_filter_on_labels(lbl: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Keep labels whose max intensity >= relmax_thr * best_max; background stays 0."""
        if lbl.max() == 0:
            return lbl

        props = regionprops(lbl, intensity_image=intensity)
        if not props:
            return lbl * 0

        max_ints = np.array([p.max_intensity for p in props], dtype=np.float32)
        best = float(max_ints.max()) if max_ints.size else 0.0
        if best <= 0:
            return lbl * 0

        rel = max_ints / (best + 1e-8)
        keep = {props[i].label for i in range(len(props)) if rel[i] >= relative_max_intensity_threshold}
        if not keep:
            return lbl * 0

        out_lbl = lbl.copy()
        out_lbl[~np.isin(out_lbl, list(keep))] = 0
        return out_lbl

    # Optional relmax filtering before final labeling
    if relmax_filter_stage in {"pre", "both"}:
        cc = label(bw_merge).astype(np.int32, copy=False)
        cc = _apply_relmax_filter_on_labels(cc, img_smooth).astype(np.int32, copy=False)
        bw_merge = cc > 0

    # ---- Connected-component labeling only ----
    lbl = label(bw_merge).astype(np.int32, copy=False)

    # Drop tiny labeled regions
    if min_region_voxels and min_region_voxels > 0 and lbl.max() > 0:
        counts = np.bincount(lbl.ravel())
        small = np.where(counts < int(min_region_voxels))[0]
        small = small[(small != 0)]
        if small.size:
            lbl[np.isin(lbl, small)] = 0

    # Optional relmax filtering on final regions
    if relmax_filter_stage in {"post", "both"}:
        lbl = _apply_relmax_filter_on_labels(lbl, img_smooth).astype(np.int32, copy=False)

    if relabel_sequential and lbl.max() > 0:
        lbl, _, _ = _relabel_sequential(lbl)

    if verbose:
        LOG_.info(
            "segmentChromosomeTerritories done n_labels=%d fg_voxels=%d time_s=%.3f",
            int(lbl.max()),
            int(np.count_nonzero(lbl)),
            t.s(),
        )

    return lbl.astype(np.int32, copy=False)


def segmentSpeckles(
    array_zyx: np.ndarray,
    intensity_scaling_param: list[float] | tuple[float, ...] = (1, 12.5),
    gaussian_smoothing_sigma: float = 1.0,
    s2_param: list[list[float]] | tuple[tuple[float, float], ...] = ((4, 0.10), (2.5, 0.10), (1, 0.02)),
    apply_intensity_normalization: bool = True,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Segment speckle-like dots from a 3D fluorescence image (Z,Y,X).

    Runs dot detection slice-by-slice (2D) and returns a binary mask in the
    *same dtype* as the input: background=0, foreground=max(dtype) (or 1.0 for floats).
    """
    t = Timer()

    try:
        from aicssegmentation.core.vessel import filament_2d_wrapper
        from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
        from aicssegmentation.core.utils import hole_filling
        from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
        from skimage.morphology import remove_small_objects,  dilation, erosion, ball  
        from skimage.segmentation import watershed
    except ImportError as e:
        raise ImportError(
            "segmentSpeckles() requires `aicssegmentation` (for preprocessing utilities). "
            "Install it in the environment where you run it."
        ) from e

    if array_zyx.ndim != 3:
        raise ValueError(f"array_zyx must be 3D (Z,Y,X); got shape={array_zyx.shape}")

    orig_dtype = array_zyx.dtype

    if verbose:
        LOG.info(
            "segmentSpeckles start shape=%s dtype=%s sigma=%s apply_norm=%s s2_param=%s",
            array_zyx.shape,
            orig_dtype,
            gaussian_smoothing_sigma,
            apply_intensity_normalization,
            s2_param,
        )

    img = array_zyx.astype(np.float32, copy=False)

    if apply_intensity_normalization:
        img = intensity_normalization(img, scaling_param=intensity_scaling_param)

    img_smooth = image_smoothing_gaussian_3d(img, sigma=gaussian_smoothing_sigma)

    # Dot detection (slice-by-slice 2D wrapper)
    bw_spot = dot_2d_slice_by_slice_wrapper(img_smooth, s2_param)

    # Ensure boolean mask
    bw_spot = bw_spot > 0

    # Convert boolean mask to same dtype as input
    if np.issubdtype(orig_dtype, np.integer):
        fg_val = np.iinfo(orig_dtype).max
    elif np.issubdtype(orig_dtype, np.floating):
        fg_val = np.array(1.0, dtype=orig_dtype)
    else:
        fg_val = 1

    out = np.zeros_like(array_zyx, dtype=orig_dtype)
    out[bw_spot] = fg_val

    if verbose:
        n_fg = int(np.count_nonzero(out))
        LOG.info("segmentSpeckles done fg_voxels=%s time_s=%.3f", n_fg, t.s())

    return out


def segmentTelomeres(
    array_zyx: np.ndarray,
    intensity_scaling_param: list[float] | tuple[float, ...] = [1.5, 27],
    gaussian_smoothing_sigma: float = 1.0,
    s2_param: list[list[float]] | tuple[tuple[float, float], ...] = [[1.5,0.06], [1,0.04]],
    apply_intensity_normalization: bool = True,
    min_region_max_intensity: float = 350.0,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Segment telomere-like dots from a 3D fluorescence image (Z,Y,X).

    Baseline: same as `segmentSpeckles` (intensity normalization -> 3D Gaussian smoothing
    -> dot detection slice-by-slice). Modification: applies an object-level filter so
    only connected spot regions whose *maximum raw intensity* is >= `min_region_max_intensity`
    are kept.

    Returns a binary mask in the *same dtype* as the input: background=0,
    foreground=max(dtype) (or 1.0 for floats).

    Parameters
    ----------
    array_zyx : np.ndarray
        3D input image with shape (Z, Y, X).
    intensity_scaling_param : list[float] | tuple[float, ...]
        Parameters passed to `intensity_normalization` when `apply_intensity_normalization` is True.
    gaussian_smoothing_sigma : float
        Sigma for 3D Gaussian smoothing (applied to the working float32 image).
    s2_param : list[list[float]] | tuple[tuple[float, float], ...]
        Parameters passed to AICS `dot_2d_slice_by_slice_wrapper`.
    apply_intensity_normalization : bool
        Whether to apply AICS `intensity_normalization` to the working image before detection.
    min_region_max_intensity : float
        Minimum *max intensity in the raw input image* required for a connected spot region
        to be kept (telomeres are spots).
    verbose : bool
        If True, logs basic info (requires your existing `LOG` and `Timer`).

    Returns
    -------
    np.ndarray
        Binary mask in same dtype/shape as input.
    """
    t = Timer()

    try:
        from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
        from aicssegmentation.core.pre_processing_utils import (
            intensity_normalization,
            image_smoothing_gaussian_3d,
        )
        from skimage.measure import label, regionprops
    except ImportError as e:
        raise ImportError(
            "segmentTelomeres() requires `aicssegmentation` and `scikit-image`."
        ) from e

    if array_zyx.ndim != 3:
        raise ValueError(f"array_zyx must be 3D (Z,Y,X); got shape={array_zyx.shape}")

    orig_dtype = array_zyx.dtype

    if verbose:
        LOG.info(
            "segmentTelomeres start shape=%s dtype=%s sigma=%s apply_norm=%s s2_param=%s min_region_max_intensity=%s",
            array_zyx.shape,
            orig_dtype,
            gaussian_smoothing_sigma,
            apply_intensity_normalization,
            s2_param,
            min_region_max_intensity,
        )

    # --- preprocessing / detection (same baseline as segmentSpeckles) ---
    img = array_zyx.astype(np.float32, copy=False)

    if apply_intensity_normalization:
        img = intensity_normalization(img, scaling_param=intensity_scaling_param)

    img_smooth = image_smoothing_gaussian_3d(img, sigma=gaussian_smoothing_sigma)

    bw_spot = dot_2d_slice_by_slice_wrapper(img_smooth, s2_param) > 0  # boolean

    # --- object-level max-intensity filter on RAW image ---
    # Use the original signal (not normalized) for thresholding.
    raw_float = array_zyx.astype(np.float32, copy=False)

    lab = label(bw_spot, connectivity=1)
    if lab.max() > 0:
        keep = np.zeros(lab.max() + 1, dtype=bool)  # index 0 unused (background)
        # regionprops with intensity_image gives max_intensity per region
        for rp in regionprops(lab, intensity_image=raw_float):
            keep[rp.label] = (rp.max_intensity >= float(min_region_max_intensity))
        bw_spot = keep[lab]  # vectorized remap labels->keep
    else:
        bw_spot = np.zeros_like(bw_spot, dtype=bool)

    # --- convert boolean mask to same dtype as input ---
    if np.issubdtype(orig_dtype, np.integer):
        fg_val = np.iinfo(orig_dtype).max
    elif np.issubdtype(orig_dtype, np.floating):
        fg_val = np.array(1.0, dtype=orig_dtype)
    else:
        fg_val = 1

    out = np.zeros_like(array_zyx, dtype=orig_dtype)
    out[bw_spot] = fg_val

    if verbose:
        n_lbl = int(lab.max())
        n_fg = int(np.count_nonzero(out))
        LOG.info(
            "segmentTelomeres done n_regions=%s fg_voxels=%s time_s=%.3f",
            n_lbl,
            n_fg,
            t.s(),
        )

    return out