# coding: utf-8
# Author: Miiko Sokka

import numpy as np

def segmentMetaphaseChromosomes(array_3D, f3_param = [[1, 0.01]], minArea = 4):
    """
    Segment chromosome-like filaments from a 3D fluorescence image using a filament detection algorithm.

    Parameters
    ----------
    array_3D : np.ndarray
        A 3D NumPy array of shape (Z, Y, X), expected to be histogram-stretched and of type uint16.
    f3_param : list of list, optional
        Parameters for the `filament_3d_wrapper` function, controlling filament detection.
        Default is [[1, 0.01]]. This works well for metaphase chromosome spreads
    minArea : int, optional
        Minimum size (in voxels) for an object to be retained in the post-processing step.
        Default is 4.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of the same shape as `array_3D`, of type uint16.
        Values are 0 for background and 65535 for segmented chromosome-like objects.
    """

    try:
        from aicssegmentation.core.vessel import filament_3d_wrapper
        from skimage.morphology import remove_small_objects
    except ImportError as e:
        raise ImportError(
            "find_centroids() requires `aicssegmentation` to be installed.\n"
            "Install it in the environment where you run centroid detection."
        ) from e

    # Segmentation
    bw = filament_3d_wrapper(array_3D, f3_param)

    # Post processing
    seg = remove_small_objects(bw>0, min_size=minArea, connectivity=1)
    
    seg_uint16 = (seg * 65535).astype(np.uint16)

    return seg_uint16


def segmentChromosomeTerritories(
    array_zyx: np.ndarray,
    intensity_scaling_param=(0.5, 15),
    gaussian_smoothing_sigma: float = 1.0,
    s2_param_bright=None,
    s2_param_dark=None,
    threshold: str = "ave",
    min_area: int = 700,
    apply_intensity_normalization: bool = True,
) -> np.ndarray:
    """
    Segment chromosome territories from a 3D fluorescence image using the
    AICS-segmentation Masked-Object (MO) thresholding and 2D dot filters.

    Parameters
    ----------
    array_zyx : np.ndarray
        3D input image with shape (Z, Y, X). Typically a histogram-stretched
        fluorescence image (e.g. uint8 or uint16).
    intensity_scaling_param : tuple or list, optional
        Parameters passed to `intensity_normalization` if
        `apply_intensity_normalization` is True. Default is (0.5, 15).
    gaussian_smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing in voxels.
        Passed to `image_smoothing_gaussian_3d`. Default is 1.0.
    s2_param_bright : list of [sigma, threshold], optional
        Parameters for bright-spot detection via
        `dot_2d_slice_by_slice_wrapper`. If None, defaults to [[2, 0.025]].
    s2_param_dark : list of [sigma, threshold], optional
        Parameters for dark-spot detection (on 1 - image) via
        `dot_2d_slice_by_slice_wrapper`. If None, defaults to
        [[2, 0.025], [1, 0.025]].
    threshold : str or float, optional
        Global thresholding method passed to `MO`. Default is "ave".
    min_area : int, optional
        Minimum object size (in voxels) for MO thresholding.
        Passed to `MO` as `object_minArea`. Default is 700.
    apply_intensity_normalization : bool, optional
        If True, apply `intensity_normalization` with `intensity_scaling_param`
        before smoothing and segmentation. If False, the input is only cast to
        float32 and then smoothed.

    Returns
    -------
    np.ndarray
        A 3D array of the same shape as `array_zyx`, with the same dtype as
        the input. Background voxels are 0, and foreground (segmented
        chromosome territories) are set to the maximum value of the input dtype
        (e.g. 255 for uint8, 65535 for uint16, 1.0 for float images).
    """
    try:
        from aicssegmentation.core.MO_threshold import MO
        from aicssegmentation.core.pre_processing_utils import (
            intensity_normalization,
            image_smoothing_gaussian_3d,
        )
        from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
        from skimage.morphology import remove_small_objects
    except ImportError as e:
        raise ImportError(
            "segmentChromosomeTerritories() requires `aicssegmentation` to be "
            "installed (and its dependencies such as scikit-image). "
            "Install it in the environment where you run this function."
        ) from e

    if array_zyx.ndim != 3:
        raise ValueError(
            f"array_zyx must be 3D with shape (Z, Y, X); got shape {array_zyx.shape}"
        )

    # Defaults for dot filters
    if s2_param_bright is None:
        s2_param_bright = [[2, 0.025]]
    if s2_param_dark is None:
        s2_param_dark = [[2, 0.025], [1, 0.025]]

    # Keep track of original dtype for output scaling
    orig_dtype = array_zyx.dtype

    # Work in float32
    img = array_zyx.astype(np.float32, copy=False)

    # Optional intensity normalization
    if apply_intensity_normalization:
        img = intensity_normalization(img, scaling_param=intensity_scaling_param)

    # Smoothing
    img_smooth = image_smoothing_gaussian_3d(img, sigma=gaussian_smoothing_sigma,)

    # Masked-Object (MO) Thresholding
    bw_mo, _ = MO(
        img_smooth,
        global_thresh_method=threshold,
        object_minArea=min_area,
        return_object=True,
    )
    
    # For now, the segmentation works better if the dot slice function is skipped
    
    # # 2D bright-spot filtering
    # bw_extra = dot_2d_slice_by_slice_wrapper(img_smooth, s2_param_bright)

    # 2D dark-spot filtering on inverted image
    # bw_dark = dot_2d_slice_by_slice_wrapper(1.0 - img_smooth, s2_param_dark)

    # # Merge MO + bright-spot filters, then remove dark spots
    # bw_merge = np.logical_or(bw_mo, bw_extra)
    bw_merge = bw_mo
    # bw_merge[bw_dark > 0] = False

    # Remove small objects
    bw_merge = remove_small_objects(bw_merge>0, min_size=1000, connectivity=0)

    # Convert boolean mask to same dtype as input
    if np.issubdtype(orig_dtype, np.integer):
        fg_val = np.iinfo(orig_dtype).max
    elif np.issubdtype(orig_dtype, np.floating):
        fg_val = np.array(1.0, dtype=orig_dtype)
    else:
        # Fallback: treat as integer
        fg_val = 1

    out = np.zeros_like(array_zyx, dtype=orig_dtype)
    out[bw_merge] = fg_val

    return out