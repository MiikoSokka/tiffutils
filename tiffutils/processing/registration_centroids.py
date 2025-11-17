# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import tiffutils as tiffu

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.utils import peak_local_max_wrapper

from skimage.morphology import remove_small_objects, ball, binary_dilation, dilation
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_dilation


def find_centroids(
    array_3D,
    s3_param=[[1, 0.08]],
    minArea=20,
    sphere_diameter=6,
    gaussian_smoothing_sigma=1,
):
    
    """
    Full pipeline:
      1) Normalize 3D array
      2) Segment beads
      3) Find centroids
      4) Convert centroids to a 3D boolean sphere mask

    Parameters
    ----------
    array_3D : np.ndarray
        Input 3D image (Z, Y, X).
    s3_param : list, optional
        Parameters passed to dot_3d_wrapper.
    minArea : int, optional
        Minimum object size for remove_small_objects.
    sphere_diameter : int, optional
        Approximate diameter (in voxels) of spheres drawn at centroid locations.
        Used to define the structuring element radius for dilation.
    gaussian_smoothing_sigma : float, optional
        Sigma for Gaussian smoothing (slice-by-slice).

    Returns
    -------
    centroids_mask : np.ndarray (bool)
        3D boolean mask (Z, Y, X) where True marks voxels inside centroid spheres.
    centroids_3d : np.ndarray
        Array of centroids, shape (N, 3) with (z, y, x) coordinates (float).
    """

    if array_3D.ndim != 3:
        raise ValueError(f"array_3D must be 3D (Z, Y, X). Got shape {array_3D.shape}")

    print("Detect features...")

    # 1) Normalize
    array_3D_normalized = tiffu.convert_dtype(tiffu.histogram_stretch(
        array_3D,
        intensity_scaling_param=[1, 99.99]
    ), 'float32'
                                             )

    # Optional Gaussian smoothing (slice-by-slice)
    array_3D_smooth = image_smoothing_gaussian_slice_by_slice(
        array_3D_normalized,
        sigma=gaussian_smoothing_sigma,
    )
    
    # 2) Segment dots
    print("Segment dots...")
    print(f"\tS3 param {s3_param}")

    # Segmentation from your custom 3D dot detector
    bw = dot_3d_wrapper(array_3D_smooth, s3_param)  # expected to be mask-like

    # Ensure boolean
    bw = bw > 0

    # Remove small objects
    mask = remove_small_objects(bw, min_size=minArea, connectivity=1)

    # If nothing remains, return early with empty results
    if not mask.any():
        print("No objects found after size filtering; returning empty results.")
        centroids_3d = np.zeros((0, 3), dtype=float)
        centroids_mask = np.zeros_like(mask, dtype=bool)        
        return centroids_mask, centroids_3d

    # Seeds for watershed
    seeds = dilation(
        peak_local_max_wrapper(array_3D_smooth, label(mask))
    )

    # Watershed on distance transform of bw
    watershed_map = -distance_transform_edt(bw)
    labels_ws = watershed(
        watershed_map,
        label(seeds),
        mask=mask,
        watershed_line=True,
    )

    # Final binary mask and re-label for clean connected components
    final_mask = labels_ws > 0
    labeled_mask = label(final_mask)

    # 3) Define centroids
    print("Define centroids...")
    props = regionprops(labeled_mask)

    if not props:
        print("No labeled regions found; returning empty results.")
        centroids_3d = np.zeros((0, 3), dtype=float)
        centroids_mask = np.zeros_like(final_mask, dtype=bool)
        return centroids_mask, centroids_3d

    centroids_3d = np.array([p.centroid for p in props], dtype=float)  # (N, 3)

    # 4) Convert centroids to a boolean sphere mask
    print("Convert centroids into boolean image array...")
    shape = final_mask.shape
    points = np.zeros(shape, dtype=bool)

    # Mark centroid voxels
    for zf, yf, xf in centroids_3d:
        z, y, x = np.round([zf, yf, xf]).astype(int)
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            points[z, y, x] = True

    # Sphere dilation around centroids
    if sphere_diameter is None or sphere_diameter <= 1:
        # No dilation, just single voxels
        centroids_mask = points
    else:
        radius = max(1, sphere_diameter // 2)
        selem = ball(radius)
        centroids_mask = binary_dilation(points, selem)
    
    # centroids_mask is bool, ideal for phase_cross_correlation
    return centroids_mask, centroids_3d

import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift


def register_3d_stack(
    fixed_centroids: np.ndarray,
    moving_centroids: np.ndarray,
    upsample_factor: int = 1,
):
    """
    Register a 3D stack using centroid images and apply the shift to the full stack.

    Parameters
    ----------
    fixed_centroids, moving_centroids : np.ndarray
        Arrays with shape (Z, Y, X) or (Z, C, Y, X).
        Used only to estimate the 3D shift via phase_cross_correlation.
    fixed_original, moving_original : np.ndarray
        Original image stacks with the same layout as the centroid arrays:
        either (Z, Y, X) or (Z, C, Y, X).
        The computed (dz, dy, dx) shift is applied to `moving_original`.
    upsample_factor : int, optional
        Passed to skimage.registration.phase_cross_correlation for subpixel
        accuracy (default 1 = pixel accuracy).

    Returns
    -------
    registered : np.ndarray
        `moving_original` shifted into the reference frame of `fixed_original`,
        same shape and dtype as `moving_original`.
    shift_vec : np.ndarray
        The estimated shift as (dz, dy, dx).
    error : float
        Error returned by phase_cross_correlation.
    diffphase : float
        Global phase shift returned by phase_cross_correlation.
    """
    # --- Prepare 3D volumes (Z, Y, X) for shift estimation ---
    if fixed_centroids.ndim == 4:
        # Collapse channels (e.g. by max projection) to get ZYX
        fixed_for_reg = fixed_centroids.max(axis=1)   # (Z, Y, X)
        moving_for_reg = moving_centroids.max(axis=1) # (Z, Y, X)
    else:
        # Already (Z, Y, X)
        fixed_for_reg = fixed_centroids
        moving_for_reg = moving_centroids

    # --- Compute 3D shift ---
    shift_vec, error, diffphase = phase_cross_correlation(
        fixed_for_reg,
        moving_for_reg,
        upsample_factor=upsample_factor,
    )
    # shift_vec is (dz, dy, dx) for arrays shaped (Z, Y, X)
    dz, dy, dx = shift_vec

    # --- Apply shift to moving_original ---
    orig_dtype = moving_original.dtype

    if moving_original.ndim == 3:
        # Shape: (Z, Y, X)
        shift_full = (dz, dy, dx)
    else:
        # Shape: (Z, C, Y, X) -> do not shift channels
        # axes = (Z, C, Y, X) -> (shift_z, 0, shift_y, shift_x)
        shift_full = (dz, 0.0, dy, dx)

    shifted = nd_shift(
        moving_original,
        shift=shift_full,
        order=1,         # linear interpolation
        mode="constant",
        cval=0.0,
    )

    # --- Cast back to original dtype (if integer) ---
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        shifted = np.clip(np.rint(shifted), info.min, info.max).astype(orig_dtype)
    else:
        shifted = shifted.astype(orig_dtype, copy=False)

    return shifted, shift_vec, error, diffphase