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

from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)

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
        raise ValueError(
            LOG.error("array_3D must be 3D (Z, Y, X). Got shape %s", array_3D.shape)
        )

    t = Timer()
    LOG.info(
        "start step=find_centroids shape=%s dtype=%s gaussian_sigma=%.3g minArea=%d sphere_diameter=%s s3_param=%s",
        array_3D.shape,
        array_3D.dtype,
        float(gaussian_smoothing_sigma),
        int(minArea),
        sphere_diameter,
        s3_param,
    )

    # ------------------------------------------------------------------
    # 1) Normalize
    # ------------------------------------------------------------------
    array_3D_normalized = tiffu.convert_dtype(
        tiffu.histogram_stretch(array_3D, intensity_scaling_param=[1, 99.99]),
        "float32",
    )

    # Optional Gaussian smoothing (slice-by-slice)
    array_3D_smooth = image_smoothing_gaussian_slice_by_slice(
        array_3D_normalized,
        sigma=gaussian_smoothing_sigma,
    )

    # ------------------------------------------------------------------
    # 2) Segment dots
    # ------------------------------------------------------------------
    LOG.debug("step=dot_3d_wrapper s3_param=%s", s3_param)
    bw = dot_3d_wrapper(array_3D_smooth, s3_param)
    bw = bw > 0

    mask = remove_small_objects(bw, max_size=int(minArea), connectivity=1)

    if not mask.any():
        LOG.warning(
            "step=size_filter no_objects_after_filter minArea=%d bw_nonzero=%d",
            int(minArea),
            int(np.count_nonzero(bw)),
        )
        centroids_3d = np.zeros((0, 3), dtype=float)
        centroids_mask = np.zeros_like(mask, dtype=bool)
        LOG.info("done step=find_centroids n_centroids=0 time_s=%.3f", t.s())
        return centroids_mask, centroids_3d

    # ------------------------------------------------------------------
    # Watershed separation
    # ------------------------------------------------------------------
    seeds = dilation(
        peak_local_max_wrapper(array_3D_smooth, label(mask))
    )

    if not seeds.any():
        LOG.warning(
            "step=seeds no_seeds_found mask_nonzero=%d",
            int(np.count_nonzero(mask)),
        )

    watershed_map = -distance_transform_edt(bw)
    labels_ws = watershed(
        watershed_map,
        label(seeds),
        mask=mask,
        watershed_line=True,
    )

    final_mask = labels_ws > 0
    labeled_mask = label(final_mask)

    # ------------------------------------------------------------------
    # 3) Define centroids
    # ------------------------------------------------------------------
    props = regionprops(labeled_mask)

    if not props:
        LOG.warning("step=regionprops no_regions_found")
        centroids_3d = np.zeros((0, 3), dtype=float)
        centroids_mask = np.zeros_like(final_mask, dtype=bool)
        LOG.info("done step=find_centroids n_centroids=0 time_s=%.3f", t.s())
        return centroids_mask, centroids_3d

    centroids_3d = np.array([p.centroid for p in props], dtype=float)
    n_centroids = int(centroids_3d.shape[0])

    LOG.info(
        "step=centroids n_centroids=%d labels=%d",
        n_centroids,
        int(labeled_mask.max()),
    )

    # ------------------------------------------------------------------
    # 4) Convert centroids to boolean sphere mask
    # ------------------------------------------------------------------
    shape = final_mask.shape
    points = np.zeros(shape, dtype=bool)

    for zf, yf, xf in centroids_3d:
        z, y, x = np.round([zf, yf, xf]).astype(int)
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            points[z, y, x] = True

    if sphere_diameter is None or sphere_diameter <= 1:
        centroids_mask = points
        radius = 0
    else:
        radius = max(1, int(sphere_diameter) // 2)
        centroids_mask = binary_dilation(points, ball(radius))

    LOG.info(
        "done step=find_centroids n_centroids=%d radius=%d time_s=%.3f",
        n_centroids,
        int(radius),
        t.s(),
    )

    return centroids_mask, centroids_3d



import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree
from skimage.feature import match_template
from skimage.transform import warp
import logging

log = logging.getLogger(__name__)


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


def _dice_error_region_after_shift(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    shift_vec_zyx,
    region_yx,  # (y_fixed, x_fixed, h, w)
):
    dz, dy, dx = shift_vec_zyx
    y, x, h, w = region_yx

    moving_shifted = nd_shift(
        moving_mask.astype(np.float32),
        shift=(dz, dy, dx),
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    f = fixed_mask[:, y : y + h, x : x + w]
    m = moving_shifted[:, y : y + h, x : x + w]

    inter = np.count_nonzero(f & m)
    na = np.count_nonzero(f)
    nb = np.count_nonzero(m)
    if (na + nb) == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / (na + nb)
    return 1.0 - float(dice)


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


def _estimate_dz_from_z_profiles(fixed_mask: np.ndarray, moving_mask: np.ndarray, max_abs_shift: int | None = None):
    f = fixed_mask.astype(np.bool_, copy=False)
    m = moving_mask.astype(np.bool_, copy=False)

    fz = f.sum(axis=(1, 2)).astype(np.float32)
    mz = m.sum(axis=(1, 2)).astype(np.float32)

    fz -= fz.mean() if fz.size else 0.0
    mz -= mz.mean() if mz.size else 0.0

    if np.all(fz == 0) or np.all(mz == 0):
        return 0.0

    corr = np.correlate(fz, mz, mode="full")
    lags = np.arange(-(mz.size - 1), fz.size, dtype=np.int64)

    if max_abs_shift is not None:
        max_abs_shift = int(max_abs_shift)
        keep = (lags >= -max_abs_shift) & (lags <= max_abs_shift)
        corr = corr[keep]
        lags = lags[keep]

    dz = float(lags[int(np.argmax(corr))])
    return dz


def _normxcorr_match_location(fixed_2d: np.ndarray, template_2d: np.ndarray):
    """
    Normalized cross-correlation template matching.
    Returns (y, x, score) where (y,x) is the TOP-LEFT placement of template in fixed.
    """
    fixed_2d = fixed_2d.astype(np.float32, copy=False)
    template_2d = template_2d.astype(np.float32, copy=False)

    if template_2d.shape[0] > fixed_2d.shape[0] or template_2d.shape[1] > fixed_2d.shape[1]:
        return None, None, float("-inf")

    cc = match_template(fixed_2d, template_2d, pad_input=False)
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    y, x = int(peak[0]), int(peak[1])
    score = float(cc[peak])
    return y, x, score


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


def _try_sitk_affine_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, n_iter: int = 200):
    try:
        import SimpleITK as sitk
    except ImportError:
        LOG.warning("SimpleITK not available: skipping affine and returning rigid-only.")
        return None, False

    fixed = sitk.GetImageFromArray(fixed_2d.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_2d.astype(np.float32))

    tx0 = sitk.AffineTransform(2)

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(tx0, inPlace=False)
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


def _try_sitk_bspline_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, mesh_size=(2, 2), n_iter: int = 60):
    """
    VERY SMOOTH residual: coarse B-spline transform on 2D masks.
    mesh_size is intentionally tiny (2x2 by default) to only allow gentle bending.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        LOG.warning("SimpleITK not available: skipping smooth nonrigid refinement.")
        return None, False

    fixed = sitk.GetImageFromArray(fixed_2d.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_2d.astype(np.float32))

    tx0 = sitk.BSplineTransformInitializer(fixed, mesh_size)

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(tx0, inPlace=False)
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
    affine_iters: int = 200,
    smooth_bspline_mesh_size: tuple[int, int] = (2, 2),
    smooth_bspline_iters: int = 60,
):
    """
    Workflow:
      1) Estimate dz from Z-profiles (no PCC).
      2) Iterate shrinking central crop sizes:
           - NCC template match moving_crop_2d within fixed_2d
           - Convert match position -> (dy,dx)
           - Compute Dice error ONLY over the matched region
           - LOG Dice error each iteration
           - STOP EARLY once crop_dice_error <= threshold
      3) Apply that shift to FULL moving_original.
      4) Global 2D affine registration (SimpleITK) on masks; apply slice-by-slice to FULL moving_original.
      5) If (and only if) the crop-search actually *accepted* a crop (crop_dice_error <= threshold),
         run an EXTRA **very smooth** 2D B-spline refinement on top of affine; apply slice-by-slice.

    Returns
    -------
    registered : np.ndarray
        Final aligned moving_original.
    shift_vec : np.ndarray
        Rigid shift used (dz,dy,dx) before affine/nonrigid.
    error : float
        Final Dice error after the applied transforms (full volume).
    """
    if fixed_centroids.ndim == 4:
        fixed_for_reg = fixed_centroids.any(axis=1)
        moving_for_reg = moving_centroids.any(axis=1)
    else:
        fixed_for_reg = fixed_centroids.astype(bool, copy=False)
        moving_for_reg = moving_centroids.astype(bool, copy=False)

    orig_dtype = moving_original.dtype
    Z, Y, X = fixed_for_reg.shape

    dz0 = _estimate_dz_from_z_profiles(fixed_for_reg, moving_for_reg)

    fixed_2d = fixed_for_reg.any(axis=0).astype(np.float32)

    best_shift = np.asarray([dz0, 0.0, 0.0], dtype=np.float32)
    best_crop_err = np.inf
    accepted = False

    size = int(central_start)
    while size >= int(min_central):
        moving_crop_zyx, (y0, y1, x0, x1) = _center_crop_xy(moving_for_reg, size)
        h = int(y1 - y0)
        w = int(x1 - x0)

        templ_2d = moving_crop_zyx.any(axis=0).astype(np.float32)
        if np.count_nonzero(templ_2d) == 0:
            LOG.info(
                "[normxcorr] size=%4d crop(y=%d:%d, x=%d:%d) template_empty -> skip (crop_dice_error=inf)",
                size, y0, y1, x0, x1,
            )
            size -= int(central_step)
            continue

        y_peak, x_peak, score = _normxcorr_match_location(fixed_2d, templ_2d)
        if y_peak is None:
            LOG.info(
                "[normxcorr] size=%4d crop(y=%d:%d, x=%d:%d) template_larger_than_fixed -> skip (crop_dice_error=inf)",
                size, y0, y1, x0, x1,
            )
            size -= int(central_step)
            continue

        dy = float(y_peak - y0)
        dx = float(x_peak - x0)
        shift_vec = np.asarray([dz0, dy, dx], dtype=np.float32)

        region = (y_peak, x_peak, h, w)
        crop_err = _dice_error_region_after_shift(fixed_for_reg, moving_for_reg, shift_vec, region)

        LOG.info(
            "[normxcorr] size=%4d crop(y=%d:%d, x=%d:%d) match(top-left)=(y=%d,x=%d) score=%.4f "
            "shift(dz,dy,dx)=(%.2f,%.2f,%.2f) crop_dice_error=%.4f",
            size, y0, y1, x0, x1, y_peak, x_peak, score, dz0, dy, dx, crop_err,
        )

        best_shift = shift_vec
        best_crop_err = crop_err

        if crop_err <= float(dice_threshold):
            accepted = True
            break

        size -= int(central_step)

    dz, dy, dx = map(float, best_shift)
    if accepted:
        LOG.info(
            "[normxcorr] ACCEPTED: shift(dz,dy,dx)=(%.2f,%.2f,%.2f) crop_dice_error=%.4f <= %.3f",
            dz, dy, dx, best_crop_err, float(dice_threshold),
        )
    else:
        LOG.info(
            "[normxcorr] NOT ACCEPTED: using last tried shift(dz,dy,dx)=(%.2f,%.2f,%.2f) crop_dice_error=%.4f (threshold=%.3f)",
            dz, dy, dx, best_crop_err, float(dice_threshold),
        )

    shift_full = (dz, dy, dx) if moving_original.ndim == 3 else (dz, 0.0, dy, dx)
    rigid_shifted = nd_shift(
        moving_original,
        shift=shift_full,
        order=1,
        mode="constant",
        cval=0.0,
    )

    _, moving_shifted_mask = _dice_error(fixed_for_reg, moving_for_reg, best_shift)

    fixed_aff_2d = fixed_for_reg.any(axis=0).astype(np.float32)
    moving_aff_2d = moving_shifted_mask.any(axis=0).astype(np.float32)

    tx_aff, ok_aff = _try_sitk_affine_2d(fixed_aff_2d, moving_aff_2d, n_iter=int(affine_iters))

    if ok_aff:
        import SimpleITK as sitk

        def _resample_2d(arr2d: np.ndarray, tx, interp):
            mov = sitk.GetImageFromArray(arr2d.astype(np.float32))
            ref = sitk.GetImageFromArray(np.zeros_like(arr2d, dtype=np.float32))
            res = sitk.Resample(mov, ref, tx, interp, 0.0, sitk.sitkFloat32)
            return sitk.GetArrayFromImage(res)

        if rigid_shifted.ndim == 3:
            out_aff = np.empty_like(rigid_shifted, dtype=np.float32)
            for z in range(rigid_shifted.shape[0]):
                out_aff[z] = _resample_2d(rigid_shifted[z], tx_aff, sitk.sitkLinear)
        else:
            out_aff = np.empty_like(rigid_shifted, dtype=np.float32)
            for z in range(rigid_shifted.shape[0]):
                for c in range(rigid_shifted.shape[1]):
                    out_aff[z, c] = _resample_2d(rigid_shifted[z, c], tx_aff, sitk.sitkLinear)

        moving_mask_aff = np.zeros_like(moving_shifted_mask, dtype=bool)
        for z in range(moving_shifted_mask.shape[0]):
            moving_mask_aff[z] = _resample_2d(
                moving_shifted_mask[z].astype(np.float32),
                tx_aff,
                sitk.sitkNearestNeighbor,
            ) > 0.5
    else:
        out_aff = rigid_shifted.astype(np.float32, copy=False)
        moving_mask_aff = moving_shifted_mask

    if accepted and ok_aff:
        fixed_nr_2d = fixed_for_reg.any(axis=0).astype(np.float32)
        moving_nr_2d = moving_mask_aff.any(axis=0).astype(np.float32)

        tx_nr, ok_nr = _try_sitk_bspline_2d(
            fixed_nr_2d,
            moving_nr_2d,
            mesh_size=tuple(map(int, smooth_bspline_mesh_size)),
            n_iter=int(smooth_bspline_iters),
        )

        if ok_nr:
            import SimpleITK as sitk

            def _resample_2d(arr2d: np.ndarray, tx, interp):
                mov = sitk.GetImageFromArray(arr2d.astype(np.float32))
                ref = sitk.GetImageFromArray(np.zeros_like(arr2d, dtype=np.float32))
                res = sitk.Resample(mov, ref, tx, interp, 0.0, sitk.sitkFloat32)
                return sitk.GetArrayFromImage(res)

            LOG.info(
                "[smooth-nonrigid] running coarse B-spline refinement mesh=%s iters=%d",
                str(smooth_bspline_mesh_size), int(smooth_bspline_iters),
            )

            if out_aff.ndim == 3:
                out = np.empty_like(out_aff, dtype=np.float32)
                for z in range(out_aff.shape[0]):
                    out[z] = _resample_2d(out_aff[z], tx_nr, sitk.sitkLinear)
            else:
                out = np.empty_like(out_aff, dtype=np.float32)
                for z in range(out_aff.shape[0]):
                    for c in range(out_aff.shape[1]):
                        out[z, c] = _resample_2d(out_aff[z, c], tx_nr, sitk.sitkLinear)

            moving_mask_final = np.zeros_like(moving_mask_aff, dtype=bool)
            for z in range(moving_mask_aff.shape[0]):
                moving_mask_final[z] = _resample_2d(
                    moving_mask_aff[z].astype(np.float32),
                    tx_nr,
                    sitk.sitkNearestNeighbor,
                ) > 0.5
        else:
            out = out_aff
            moving_mask_final = moving_mask_aff
    else:
        out = out_aff
        moving_mask_final = moving_mask_aff

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        registered = np.clip(np.rint(out), info.min, info.max).astype(orig_dtype)
    else:
        registered = out.astype(orig_dtype, copy=False)

    inter = np.count_nonzero(fixed_for_reg & moving_mask_final)
    na = np.count_nonzero(fixed_for_reg)
    nb = np.count_nonzero(moving_mask_final)
    if (na + nb) == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / (na + nb)
    final_error = 1.0 - float(dice)

    LOG.info("done step=error=%.4f", final_error)
    return registered, np.asarray([dz, dy, dx], dtype=np.float32), float(final_error)