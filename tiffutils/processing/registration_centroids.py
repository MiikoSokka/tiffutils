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
    mask = remove_small_objects(bw, max_size=minArea, connectivity=1)

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
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from skimage.transform import warp


def _center_crop_xy(vol_zyx: np.ndarray, size: int):
    z, y, x = vol_zyx.shape
    size = int(min(size, y, x))
    cy, cx = y // 2, x // 2
    half = size // 2
    y0, y1 = max(0, cy - half), min(y, cy - half + size)
    x0, x1 = max(0, cx - half), min(x, cx - half + size)
    return vol_zyx[:, y0:y1, x0:x1], (y0, y1, x0, x1)


def _dice_error(fixed_mask: np.ndarray, moving_mask: np.ndarray, shift_vec_zyx):
    dz, dy, dx = shift_vec_zyx
    moving_shifted = nd_shift(
        moving_mask.astype(np.float32),
        shift=(dz, dy, dx),
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    inter = np.count_nonzero(fixed_mask & moving_shifted)
    na = np.count_nonzero(fixed_mask)
    nb = np.count_nonzero(moving_shifted)
    if (na + nb) == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / (na + nb)
    return 1.0 - float(dice), moving_shifted


def _extract_points_2d(mask_zyx: np.ndarray):
    m2 = mask_zyx.any(axis=0)
    yy, xx = np.nonzero(m2)
    return np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)  # (x,y)


def _mutual_nn_matches(a_xy: np.ndarray, b_xy: np.ndarray, max_dist: float):
    if a_xy.size == 0 or b_xy.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    tree_b = cKDTree(b_xy)
    d_ab, j = tree_b.query(a_xy, k=1)
    ok = d_ab <= max_dist
    a1 = a_xy[ok]
    j1 = j[ok]
    b1 = b_xy[j1]

    tree_a = cKDTree(a_xy)
    d_ba, i = tree_a.query(b1, k=1)
    mutual = (i == np.where(ok)[0]) & (d_ba <= max_dist)

    return a1[mutual], b1[mutual]


def _fit_brown_conrady_forward(fixed_xy: np.ndarray, moving_xy: np.ndarray, cx: float, cy: float, scale: float):
    if fixed_xy.shape[0] < 30:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    xu = (fixed_xy[:, 0] - cx) / scale
    yu = (fixed_xy[:, 1] - cy) / scale
    xd = (moving_xy[:, 0] - cx) / scale
    yd = (moving_xy[:, 1] - cy) / scale

    r2 = xu * xu + yu * yu
    r4 = r2 * r2
    r6 = r4 * r2

    dx = xd - xu
    dy = yd - yu

    A_x = np.stack(
        [
            xu * r2,               # k1
            xu * r4,               # k2
            xu * r6,               # k3
            2.0 * xu * yu,         # p1
            (r2 + 2.0 * xu * xu),  # p2
        ],
        axis=1,
    )
    A_y = np.stack(
        [
            yu * r2,               # k1
            yu * r4,               # k2
            yu * r6,               # k3
            (r2 + 2.0 * yu * yu),  # p1
            2.0 * xu * yu,         # p2
        ],
        axis=1,
    )

    A = np.concatenate([A_x, A_y], axis=0)
    b = np.concatenate([dx, dy], axis=0)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    k1, k2, k3, p1, p2 = map(float, theta)
    return k1, k2, k3, p1, p2


def _brown_conrady_forward_map(coords_rc: np.ndarray, k1, k2, k3, p1, p2, cx, cy, scale):
    y = (coords_rc[:, 0] - cy) / scale
    x = (coords_rc[:, 1] - cx) / scale

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x * radial + x_tan
    yd = y * radial + y_tan

    col = xd * scale + cx
    row = yd * scale + cy
    return np.stack([row, col], axis=1)


def _warp2d_lens(image_2d: np.ndarray, k1, k2, k3, p1, p2, cx, cy, scale, order: int):
    def inv_map(coords_rc):
        return _brown_conrady_forward_map(coords_rc, k1, k2, k3, p1, p2, cx, cy, scale)

    return warp(
        image_2d,
        inverse_map=inv_map,
        order=order,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )


def _try_sitk_bspline_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, mesh_size=(4, 4), n_iter=80):
    try:
        import SimpleITK as sitk
    except ImportError:
        print("SimpleITK not available: skipping B-spline residual and returning lens-only.", flush=True)
        return None, False

    fixed = sitk.GetImageFromArray(fixed_2d.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_2d.astype(np.float32))

    tx = sitk.BSplineTransformInitializer(fixed, mesh_size)

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(tx, inPlace=False)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=int(n_iter),
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e+7,
    )

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    out_tx = reg.Execute(fixed, moving)
    return out_tx, True


def register_3d_stack(
    fixed_centroids: np.ndarray,
    moving_centroids: np.ndarray,
    moving_original: np.ndarray,
    *,
    dice_threshold: float = 0.8,
    central_start: int = 500,
    central_step: int = 50,
    min_central: int = 100,
    match_max_dist: float = 60.0,
    lens_center_search_px: int = 60,
    lens_center_step_px: int = 5,
    bspline_mesh_size: tuple[int, int] = (4, 4),
    bspline_iters: int = 80,
):
    """
    Policy (single threshold check):
      1) Run rigid PCC always and compute Dice error on rigid-shifted masks.
      2) If dice_error <= dice_threshold: return rigid-only.
      3) Else (dice_error > dice_threshold): do shrinking central-crop PCC to get a better rigid shift
         (no additional threshold checks), then ALWAYS run lens model + smooth residual B-spline FFD.

    Returns
    -------
    registered : np.ndarray
        Aligned moving_original (rigid-only or refined), same shape/dtype.
    shift_vec : np.ndarray
        Final rigid shift (dz, dy, dx) applied before nonrigid refinement.
    error : float
        Final Dice error after the applied transform(s).
    """
    # --- Prepare ZYX boolean volumes for registration ---
    if fixed_centroids.ndim == 4:
        fixed_for_reg = fixed_centroids.any(axis=1)
        moving_for_reg = moving_centroids.any(axis=1)
    else:
        fixed_for_reg = fixed_centroids.astype(bool, copy=False)
        moving_for_reg = moving_centroids.astype(bool, copy=False)

    fixed_f32 = fixed_for_reg.astype(np.float32)
    moving_f32 = moving_for_reg.astype(np.float32)

    # --- 1) Rigid PCC (full volume) ---
    shift_vec, _, _ = phase_cross_correlation(fixed_f32, moving_f32)
    shift_vec = np.asarray(shift_vec, dtype=np.float32)

    rigid_error, moving_shifted_mask = _dice_error(fixed_for_reg, moving_for_reg, shift_vec)

    # --- 2) If rigid is good enough, apply rigid and return ---
    dz, dy, dx = map(float, shift_vec)
    orig_dtype = moving_original.dtype

    if rigid_error <= float(dice_threshold):
        if moving_original.ndim == 3:
            shift_full = (dz, dy, dx)
        else:
            shift_full = (dz, 0.0, dy, dx)

        shifted = nd_shift(
            moving_original,
            shift=shift_full,
            order=1,
            mode="constant",
            cval=0.0,
        )

        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            shifted = np.clip(np.rint(shifted), info.min, info.max).astype(orig_dtype)
        else:
            shifted = shifted.astype(orig_dtype, copy=False)

        return shifted, np.asarray([dz, dy, dx], dtype=np.float32), float(rigid_error)

    # --- 3) Bad rigid -> ALWAYS do central-crop PCC + refinement ---
    # Central-crop loop chooses the best shift by Dice (but does not re-branch on threshold)
    best_err = rigid_error
    best_shift = shift_vec

    size = int(central_start)
    while size >= int(min_central):
        fixed_crop, _ = _center_crop_xy(fixed_f32, size)
        moving_crop, _ = _center_crop_xy(moving_f32, size)

        sv, _, _ = phase_cross_correlation(fixed_crop, moving_crop)
        sv = np.asarray(sv, dtype=np.float32)

        e, _ = _dice_error(fixed_for_reg, moving_for_reg, sv)
        if e < best_err:
            best_err = e
            best_shift = sv

        size -= int(central_step)

    shift_vec = best_shift
    dz, dy, dx = map(float, shift_vec)

    # Update shifted mask based on the chosen best shift (used for lens/bspline)
    _, moving_shifted_mask = _dice_error(fixed_for_reg, moving_for_reg, shift_vec)

    # Apply the chosen rigid shift to moving_original
    if moving_original.ndim == 3:
        shift_full = (dz, dy, dx)
    else:
        shift_full = (dz, 0.0, dy, dx)

    rigid_shifted = nd_shift(
        moving_original,
        shift=shift_full,
        order=1,
        mode="constant",
        cval=0.0,
    )

    print(
        f"Rigid -> affine/nonrigid refinement (initial dice_error={rigid_error:.3f} > {dice_threshold:.3f}; "
        f"best-central dice_error={best_err:.3f})",
        flush=True,
    )

    # --- Lens model fit from bead correspondences (2D, after improved rigid) ---
    fixed_pts = _extract_points_2d(fixed_for_reg)
    moving_pts = _extract_points_2d(moving_shifted_mask)

    m_xy, f_xy = _mutual_nn_matches(moving_pts, fixed_pts, max_dist=float(match_max_dist))
    fixed_xy = f_xy
    moving_xy = m_xy

    Z, Y, X = fixed_for_reg.shape
    scale = float(max(X, Y))
    cx0, cy0 = 0.5 * (X - 1), 0.5 * (Y - 1)

    best = None
    if lens_center_search_px and lens_center_search_px > 0 and fixed_xy.shape[0] >= 50:
        r = int(lens_center_search_px)
        step = int(max(1, lens_center_step_px))
        for dcx in range(-r, r + 1, step):
            for dcy in range(-r, r + 1, step):
                cx = cx0 + dcx
                cy = cy0 + dcy
                k1, k2, k3, p1, p2 = _fit_brown_conrady_forward(fixed_xy, moving_xy, cx, cy, scale)

                xu = (fixed_xy[:, 0] - cx) / scale
                yu = (fixed_xy[:, 1] - cy) / scale
                r2 = xu * xu + yu * yu
                r4 = r2 * r2
                r6 = r4 * r2
                radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
                yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
                pred = np.stack([xd * scale + cx, yd * scale + cy], axis=1)

                err = np.sqrt(((pred - moving_xy) ** 2).sum(axis=1))
                score = float(np.median(err))
                cand = (score, cx, cy, k1, k2, k3, p1, p2)
                if best is None or cand[0] < best[0]:
                    best = cand

        _, cx, cy, k1, k2, k3, p1, p2 = best
    else:
        cx, cy = cx0, cy0
        k1, k2, k3, p1, p2 = _fit_brown_conrady_forward(fixed_xy, moving_xy, cx, cy, scale)

    # Apply lens warp slice-by-slice
    if rigid_shifted.ndim == 3:
        lens_out = np.empty_like(rigid_shifted, dtype=np.float32)
        for z in range(rigid_shifted.shape[0]):
            lens_out[z] = _warp2d_lens(rigid_shifted[z].astype(np.float32), k1, k2, k3, p1, p2, cx, cy, scale, order=1)
    else:
        lens_out = np.empty_like(rigid_shifted, dtype=np.float32)
        for z in range(rigid_shifted.shape[0]):
            for c in range(rigid_shifted.shape[1]):
                lens_out[z, c] = _warp2d_lens(rigid_shifted[z, c].astype(np.float32), k1, k2, k3, p1, p2, cx, cy, scale, order=1)

    # Warp mask too for bspline + final QC
    moving_mask_lens = np.zeros_like(moving_shifted_mask, dtype=bool)
    for z in range(moving_shifted_mask.shape[0]):
        warped = _warp2d_lens(moving_shifted_mask[z].astype(np.float32), k1, k2, k3, p1, p2, cx, cy, scale, order=0)
        moving_mask_lens[z] = warped > 0.5

    # --- Residual very smooth B-spline (2D) ---
    fixed_2d = fixed_for_reg.any(axis=0).astype(np.float32)
    moving_2d = moving_mask_lens.any(axis=0).astype(np.float32)

    tx, ok = _try_sitk_bspline_2d(fixed_2d, moving_2d, mesh_size=bspline_mesh_size, n_iter=bspline_iters)

    if ok:
        import SimpleITK as sitk

        def _resample_2d(arr2d: np.ndarray):
            mov = sitk.GetImageFromArray(arr2d.astype(np.float32))
            ref = sitk.GetImageFromArray(np.zeros_like(arr2d, dtype=np.float32))
            res = sitk.Resample(mov, ref, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            return sitk.GetArrayFromImage(res)

        def _resample_2d_nn(arr2d: np.ndarray):
            mov = sitk.GetImageFromArray(arr2d.astype(np.float32))
            ref = sitk.GetImageFromArray(np.zeros_like(arr2d, dtype=np.float32))
            res = sitk.Resample(mov, ref, tx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
            return sitk.GetArrayFromImage(res)

        if lens_out.ndim == 3:
            out = np.empty_like(lens_out, dtype=np.float32)
            for z in range(lens_out.shape[0]):
                out[z] = _resample_2d(lens_out[z])
        else:
            out = np.empty_like(lens_out, dtype=np.float32)
            for z in range(lens_out.shape[0]):
                for c in range(lens_out.shape[1]):
                    out[z, c] = _resample_2d(lens_out[z, c])

        moving_mask_final = np.zeros_like(moving_mask_lens, dtype=bool)
        for z in range(moving_mask_lens.shape[0]):
            moving_mask_final[z] = _resample_2d_nn(moving_mask_lens[z].astype(np.float32)) > 0.5
    else:
        out = lens_out
        moving_mask_final = moving_mask_lens

    # Cast back to original dtype
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        registered = np.clip(np.rint(out), info.min, info.max).astype(orig_dtype)
    else:
        registered = out.astype(orig_dtype, copy=False)

    # Final Dice error QC
    inter = np.count_nonzero(fixed_for_reg & moving_mask_final)
    na = np.count_nonzero(fixed_for_reg)
    nb = np.count_nonzero(moving_mask_final)
    if (na + nb) == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / (na + nb)
    final_error = 1.0 - float(dice)

    return registered, np.asarray([dz, dy, dx], dtype=np.float32), float(final_error)