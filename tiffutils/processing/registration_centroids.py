# coding: utf-8
# Author: Miiko Sokka

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.utils import peak_local_max_wrapper

from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree

from skimage.feature import match_template
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, ball, binary_dilation, dilation
from skimage.segmentation import watershed
from skimage.transform import warp

import numpy as np
import tiffutils as tiffu
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
    LOG.debug(
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
    LOG.debug("step=normalization")
    array_3D_normalized = tiffu.convert_dtype(
        tiffu.histogram_stretch(array_3D, intensity_scaling_param=[1, 99.99]),
        "float32",
    )

    # Optional Gaussian smoothing (slice-by-slice)
    LOG.debug("step=smootihing sigma=%s", gaussian_smoothing_sigma)
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


#######################################
# 3D registration function with helpers
#######################################

import numpy as np
from scipy.ndimage import shift as nd_shift

try:
    from skimage.feature import match_template
except Exception:  # pragma: no cover
    match_template = None


def register_3d_stack(
    fixed_centroids: np.ndarray,
    moving_centroids: np.ndarray,
    moving_original: np.ndarray,
    *,
    dice_threshold: float = 0.5,  # NOTE: threshold is on Dice ERROR (1-Dice), per your convention
    crop: int = 500,
    affine_iters: int = 200,
    smooth_bspline_mesh_size: tuple[int, int] = (2, 2),
    smooth_bspline_iters: int = 60,
):
    """
    Pipeline (best-by-global-Dice selection):
      Stage 0: full-volume rigid (dz via Z-profiles + XY PCC on 2D projections) -> global Dice
      Stage 1: central-crop rigid re-estimation (NCC template match) -> central+global Dice
      Stage 1b: diminishing central-crop rigid loop (NCC) -> central+global Dice
      Stage 2: lens warp (optional hook; if unavailable, skipped) -> global Dice
      Stage 3: 2D affine (SimpleITK) applied slice-by-slice -> global Dice
      Stage 4: 2D B-spline residual (SimpleITK) applied slice-by-slice -> global Dice

    Returns
    -------
    registered : np.ndarray
    shift_vec : np.ndarray  (dz,dy,dx) rigid shift associated with the returned best transform
    error : float           best FULL-volume Dice error (1-dice)

    - Stage 0 / Stage 1 can early-stop if global Dice ERROR <= dice_threshold.
    - If we ever proceed into Stage 1b, we COMMIT to running Stage 1b -> Stage 2 -> Stage 3 -> Stage 4
      sequentially, each *building on the currently-accepted result*.
    - In committed mode, a stage is only ACCEPTED (committed) if it improves global Dice (i.e. reduces Dice error)
      relative to the current accepted global Dice error. No threshold-based stopping after entering Stage 1b.
    """

    LOG.debug(
        "start step=register_3d_stack dice_threshold=%s crop=%s affine_iters=%s smooth_bspline_mesh_size=%s smooth_bspline_iters=%s",
        dice_threshold,
        crop,
        affine_iters,
        smooth_bspline_mesh_size,
        smooth_bspline_iters,
    )

    # Normalize masks to (Z,Y,X) boolean
    if fixed_centroids.ndim == 4:
        fixed_mask = fixed_centroids.any(axis=1)
        moving_mask = moving_centroids.any(axis=1)
    else:
        fixed_mask = fixed_centroids.astype(bool, copy=False)
        moving_mask = moving_centroids.astype(bool, copy=False)

    orig_dtype = moving_original.dtype

    fixed_2d = fixed_mask.any(axis=0).astype(np.float32)
    moving_2d = moving_mask.any(axis=0).astype(np.float32)

    # -------------------------
    # "Current accepted" state (monotone improvement in committed mode)
    # -------------------------
    current_shift = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)  # (dz,dy,dx)
    current_reg = moving_original
    current_mask = moving_mask
    current_err = _dice_error_from_masks(fixed_mask, moving_mask)
    current_stage = "init"

    committed = False  # becomes True if we enter Stage 1b

    def _accept_if_improves(stage: str, reg_vol: np.ndarray, moved_mask: np.ndarray, shift_vec_zyx: np.ndarray):
        nonlocal current_shift, current_reg, current_mask, current_err, current_stage

        err = _dice_error_from_masks(fixed_mask, moved_mask)
        if float(err) < float(current_err):
            current_stage = stage
            current_err = float(err)
            current_shift = np.asarray(shift_vec_zyx, dtype=np.float32)
            current_reg = reg_vol
            current_mask = moved_mask
            LOG.debug(
                "accept stage=%s global_err=%.4f shift=(%.2f,%.2f,%.2f)",
                stage,
                float(err),
                float(current_shift[0]),
                float(current_shift[1]),
                float(current_shift[2]),
            )
            return True, float(err)
        else:
            LOG.debug(
                "reject stage=%s global_err=%.4f (no improvement; keep %.4f)",
                stage,
                float(err),
                float(current_err),
            )
            return False, float(err)

    # Helper crop box for central Dice checks (XY only, all Z)
    def _crop_box_xy(size: int):
        z, y, x = fixed_mask.shape
        size = int(min(size, y, x))
        cy, cx = y // 2, x // 2
        half = size // 2
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(y, y0 + size)
        x1 = min(x, x0 + size)
        return (y0, y1, x0, x1)

    def _central_dice_error(fmask: np.ndarray, mmask: np.ndarray, box):
        y0, y1, x0, x1 = box
        return _dice_error_from_masks(fmask[:, y0:y1, x0:x1], mmask[:, y0:y1, x0:x1])

    # -------------------------
    # Stage 0 — Full-volume rigid (PCC)
    # -------------------------
    dz0 = _estimate_dz_from_z_profiles(fixed_mask, moving_mask)
    dy0, dx0, pcc_score = _pcc_shift_2d(fixed_2d, moving_2d)

    shift0 = np.asarray([float(dz0), float(dy0), float(dx0)], dtype=np.float32)
    reg0 = _shift_full_volume(moving_original, shift0)
    moved0 = _shift_mask(moving_mask, shift0)

    _accept_if_improves("stage0_pcc_full", reg0, moved0, shift0)

    LOG.debug(
        "stage0 dz=%.2f dy=%.2f dx=%.2f pcc_score=%s global_err=%.4f thr=%.4f",
        float(dz0),
        float(dy0),
        float(dx0),
        str(pcc_score),
        float(current_err),
        float(dice_threshold),
    )

    if float(current_err) <= float(dice_threshold):
        out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
        LOG.info("done stage=%s error=%.4f (stop: global<=thr)", current_stage, float(current_err))
        return out, current_shift.copy(), float(current_err)

    # -------------------------
    # Stage 1 — Central-crop NCC from ORIGINAL moving position (no XY shift)
    # -------------------------
    crop_box = _crop_box_xy(int(crop))

    dy1, dx1, ncc_score1 = _xy_shift_from_ncc_template_absolute(
        fixed_2d=fixed_2d,
        moving_mask_zyx_unshifted=moving_mask,  # IMPORTANT: unshifted
        crop_box=crop_box,
    )

    # keep dz from z-profile estimate (dz0), but XY from NCC absolute
    shift1 = np.asarray([float(dz0), float(dy1), float(dx1)], dtype=np.float32)

    reg1 = _shift_full_volume(moving_original, shift1)
    moved1 = _shift_mask(moving_mask, shift1)

    _accept_if_improves("stage1_ncc_center_abs", reg1, moved1, shift1)

    err1_central = _central_dice_error(fixed_mask, moved1, crop_box)
    LOG.debug(
        "stage1 (ABS) crop=%s ncc_score=%.4f shift=(%.2f,%.2f,%.2f) central_err=%.4f current_global_err=%.4f",
        str(crop_box),
        float(ncc_score1),
        float(shift1[0]),
        float(shift1[1]),
        float(shift1[2]),
        float(err1_central),
        float(current_err),
    )

    # Decide stage transition:
    proceed_to_lens = float(err1_central) <= float(dice_threshold)

    # -------------------------
    # Stage 1 / 1b — NCC: select BEST shift by central Dice
    # -------------------------
    crop_box0 = _crop_box_xy(int(crop))

    best_ncc = {
        "shift": None,
        "central_err": float("inf"),
        "ncc_score": float("-inf"),
        "box": crop_box0,
    }

    # ---- Stage 1 (single crop) ----
    dy1, dx1, ncc_score1 = _xy_shift_from_ncc_template_absolute(
        fixed_2d=fixed_2d,
        moving_mask_zyx_unshifted=moving_mask,  # IMPORTANT: unshifted
        crop_box=crop_box0,
    )
    shift1 = np.asarray([float(dz0), float(dy1), float(dx1)], dtype=np.float32)

    central_err1, _moved1 = _central_err_for_shift(shift1, crop_box0)

    LOG.debug(
        "stage1 (ABS, central-only) crop=%s ncc_score=%.4f shift=(%.2f,%.2f,%.2f) central_err=%.4f",
        str(crop_box0),
        float(ncc_score1),
        float(shift1[0]), float(shift1[1]), float(shift1[2]),
        float(central_err1),
    )

    if float(central_err1) < float(best_ncc["central_err"]):
        best_ncc.update({"shift": shift1, "central_err": float(central_err1), "ncc_score": float(ncc_score1), "box": crop_box0})

    # Decide whether we need 1b:
    proceed_to_lens = float(central_err1) <= float(dice_threshold)

    # ---- Stage 1b (diminishing crops) ----
    if not proceed_to_lens:
        committed = True  # your requested behavior: entering 1b commits to later stages

        cur_size = int(crop)
        max_loops = 12

        for k in range(max_loops):
            cur_size = int(np.floor(cur_size * 0.7))
            if cur_size < 100:
                break

            box = _crop_box_xy(cur_size)

            dy_k, dx_k, ncc_score = _xy_shift_from_ncc_template_absolute(
                fixed_2d=fixed_2d,
                moving_mask_zyx_unshifted=moving_mask,
                crop_box=box,
            )
            shift_k = np.asarray([float(dz0), float(dy_k), float(dx_k)], dtype=np.float32)

            central_err_k, _moved_k = _central_err_for_shift(shift_k, box)

            LOG.debug(
                "stage1b (ABS, central-only) iter=%d crop_size=%d box=%s ncc_score=%.4f shift=(%.2f,%.2f,%.2f) central_err=%.4f best_central_err=%.4f",
                k + 1,
                cur_size,
                str(box),
                float(ncc_score),
                float(shift_k[0]), float(shift_k[1]), float(shift_k[2]),
                float(central_err_k),
                float(best_ncc["central_err"]),
            )

            if float(central_err_k) < float(best_ncc["central_err"]):
                best_ncc.update({"shift": shift_k, "central_err": float(central_err_k), "ncc_score": float(ncc_score), "box": box})

            if float(central_err_k) <= float(dice_threshold):
                proceed_to_lens = True
                break

        if not proceed_to_lens:
            LOG.debug("stage1b ended without central<=thr; committed mode => proceeding to later stages anyway")
            proceed_to_lens = True

    # ---- Commit the BEST NCC shift (by central Dice) as the rigid baseline ----
    if best_ncc["shift"] is None:
        # extremely rare: NCC produced no valid template matches
        LOG.warning("stage1/1b: no NCC result; keeping current state")
    else:
        best_shift = best_ncc["shift"]

        # Apply BEST NCC shift to FULL moving (this becomes baseline for later stages)
        current_shift = best_shift.copy()
        current_reg = _shift_full_volume(moving_original, current_shift)
        current_mask = _shift_mask(moving_mask, current_shift)

        # global error tracked for later stage acceptance (lens/affine/bspline)
        current_err = _dice_error_from_masks(fixed_mask, current_mask)
        current_stage = "stage1_best_ncc_by_central"

        LOG.debug(
            "stage1/1b BEST (by central) box=%s ncc_score=%.4f central_err=%.4f -> committed as baseline; global_err=%.4f shift=(%.2f,%.2f,%.2f)",
            str(best_ncc["box"]),
            float(best_ncc["ncc_score"]),
            float(best_ncc["central_err"]),
            float(current_err),
            float(current_shift[0]), float(current_shift[1]), float(current_shift[2]),
        )

    # Gate Stage2+ only if (a) central was good OR (b) we entered 1b (committed)
    if not proceed_to_lens:
        out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
        LOG.info("done stage=%s error=%.4f (stop: not committed and central<thr)", current_stage, float(current_err))
        return out, current_shift.copy(), float(current_err)

    # -------------------------
    # Stage 2 — Lens warp (optional hook)
    # -------------------------
    lens_reg, lens_mask, ok_lens = _try_lens_warp(
        fixed_mask=fixed_mask,
        moving_reg=current_reg,
        moving_mask=current_mask,
        match_max_dist=100.0,
        lens_center_search_px=100,
        lens_center_step_px=5,
    )

    if ok_lens:
        _accept_if_improves("stage2_lens", lens_reg, lens_mask, current_shift)
    else:
        LOG.debug("stage2 ok=0 (skipped)")

    if (not committed) and float(current_err) <= float(dice_threshold):
        out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
        LOG.info("done stage=%s error=%.4f (stop: global<=thr)", current_stage, float(current_err))
        return out, current_shift.copy(), float(current_err)

    # -------------------------
    # Stage 3 — Affine
    # -------------------------
    fixed_aff_2d = fixed_mask.any(axis=0).astype(np.float32)
    moving_aff_2d = current_mask.any(axis=0).astype(np.float32)

    tx_aff, ok_aff = _try_sitk_affine_2d(fixed_aff_2d, moving_aff_2d, n_iter=int(affine_iters))
    if ok_aff:
        import SimpleITK as sitk

        aff_reg = _apply_sitk_tx_slicewise(current_reg, tx_aff, sitk.sitkLinear)

        aff_mask = np.zeros_like(current_mask, dtype=bool)
        for z in range(current_mask.shape[0]):
            aff_mask[z] = (
                _sitk_resample_2d(current_mask[z].astype(np.float32), tx_aff, sitk.sitkNearestNeighbor) > 0.5
            )

        _accept_if_improves("stage3_affine", aff_reg, aff_mask, current_shift)
    else:
        LOG.debug("stage3 ok=0 (skipped/failed)")

    if (not committed) and float(current_err) <= float(dice_threshold):
        out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
        LOG.info("done stage=%s error=%.4f (stop: global<=thr)", current_stage, float(current_err))
        return out, current_shift.copy(), float(current_err)

    # -------------------------
    # Stage 4 — B-spline
    # -------------------------
    fixed_nr_2d = fixed_mask.any(axis=0).astype(np.float32)
    moving_nr_2d = current_mask.any(axis=0).astype(np.float32)

    tx_nr, ok_nr = _try_sitk_bspline_2d(
        fixed_nr_2d,
        moving_nr_2d,
        mesh_size=tuple(map(int, smooth_bspline_mesh_size)),
        n_iter=int(smooth_bspline_iters),
    )
    if ok_nr:
        import SimpleITK as sitk

        nr_reg = _apply_sitk_tx_slicewise(current_reg, tx_nr, sitk.sitkLinear)

        nr_mask = np.zeros_like(current_mask, dtype=bool)
        for z in range(current_mask.shape[0]):
            nr_mask[z] = (
                _sitk_resample_2d(current_mask[z].astype(np.float32), tx_nr, sitk.sitkNearestNeighbor) > 0.5
            )

        _accept_if_improves("stage4_bspline", nr_reg, nr_mask, current_shift)
    else:
        LOG.debug("stage4 ok=0 (skipped/failed)")

    out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
    LOG.info("done stage=%s error=%.4f committed=%s", current_stage, float(current_err), bool(committed))
    return out, current_shift.copy(), float(current_err)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _cast_like_input(arr: np.ndarray, orig_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(orig_dtype)
    return arr.astype(orig_dtype, copy=False)


def _shift_full_volume(vol: np.ndarray, shift_vec_zyx: np.ndarray) -> np.ndarray:
    dz, dy, dx = map(float, shift_vec_zyx)
    shift_full = (dz, dy, dx) if vol.ndim == 3 else (dz, 0.0, dy, dx)
    out = nd_shift(vol, shift=shift_full, order=1, mode="constant", cval=0.0)
    return out.astype(np.float32, copy=False)


def _shift_mask(mask_zyx: np.ndarray, shift_vec_zyx: np.ndarray) -> np.ndarray:
    dz, dy, dx = map(float, shift_vec_zyx)
    moved = nd_shift(
        mask_zyx.astype(np.float32, copy=False),
        shift=(dz, dy, dx),
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5
    return moved


def _dice_error_from_masks(fixed_mask: np.ndarray, moving_mask: np.ndarray) -> float:
    inter = np.count_nonzero(fixed_mask & moving_mask)
    na = np.count_nonzero(fixed_mask)
    nb = np.count_nonzero(moving_mask)
    dice = 1.0 if (na + nb) == 0 else (2.0 * inter) / (na + nb)
    return 1.0 - float(dice)


def _estimate_dz_from_z_profiles(fixed_mask: np.ndarray, moving_mask: np.ndarray, max_abs_shift: int | None = None) -> float:
    fz = fixed_mask.sum(axis=(1, 2)).astype(np.float32)
    mz = moving_mask.sum(axis=(1, 2)).astype(np.float32)

    if fz.size == 0 or mz.size == 0:
        return 0.0

    fz = fz - float(fz.mean())
    mz = mz - float(mz.mean())

    if np.all(fz == 0) or np.all(mz == 0):
        return 0.0

    corr = np.correlate(fz, mz, mode="full")
    lags = np.arange(-(mz.size - 1), fz.size, dtype=np.int64)

    if max_abs_shift is not None:
        max_abs_shift = int(max_abs_shift)
        keep = (lags >= -max_abs_shift) & (lags <= max_abs_shift)
        corr = corr[keep]
        lags = lags[keep]

    return float(lags[int(np.argmax(corr))])


def _pcc_shift_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, upsample_factor: int = 10):
    fixed_2d = fixed_2d.astype(np.float32, copy=False)
    moving_2d = moving_2d.astype(np.float32, copy=False)

    try:
        from skimage.registration import phase_cross_correlation
        shift, error, _ = phase_cross_correlation(fixed_2d, moving_2d, upsample_factor=int(upsample_factor))
        return float(shift[0]), float(shift[1]), float(error)
    except Exception:
        pass

    try:
        from skimage.feature import register_translation
        shift, error, _ = register_translation(fixed_2d, moving_2d, upsample_factor=int(upsample_factor))
        return float(shift[0]), float(shift[1]), float(error)
    except Exception as e:
        LOG.warning("PCC not available/failed (%s): using (dy,dx)=(0,0).", str(e))
        return 0.0, 0.0, None


def _xy_shift_from_ncc_template_absolute(
    *,
    fixed_2d: np.ndarray,
    moving_mask_zyx_unshifted: np.ndarray,
    crop_box: tuple[int, int, int, int],
):
    """
    NCC on a central crop template extracted from the ORIGINAL (unshifted) moving mask.

    Returns
    -------
    dy : float
    dx : float
    score : float
    """
    if match_template is None:
        LOG.warning("skimage.feature.match_template unavailable -> NCC absolute shift disabled")
        return 0.0, 0.0, float("-inf")

    y0, y1, x0, x1 = crop_box
    templ_2d = moving_mask_zyx_unshifted[:, y0:y1, x0:x1].any(axis=0).astype(np.float32)

    if np.count_nonzero(templ_2d) == 0:
        LOG.debug("ncc_template_empty box=%s -> (dy,dx)=(0,0)", str(crop_box))
        return 0.0, 0.0, float("-inf")

    fixed_2d = fixed_2d.astype(np.float32, copy=False)

    if templ_2d.shape[0] > fixed_2d.shape[0] or templ_2d.shape[1] > fixed_2d.shape[1]:
        LOG.debug("ncc_template_larger_than_fixed box=%s -> (dy,dx)=(0,0)", str(crop_box))
        return 0.0, 0.0, float("-inf")

    cc = match_template(fixed_2d, templ_2d, pad_input=False)
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    y_peak, x_peak = int(peak[0]), int(peak[1])
    score = float(cc[peak])

    # peak is TOP-LEFT placement of template in fixed
    dy = float(y_peak - y0)
    dx = float(x_peak - x0)
    return dy, dx, score


def _try_lens_warp(
    *,
    fixed_mask: np.ndarray,
    moving_reg: np.ndarray,
    moving_mask: np.ndarray,
    match_max_dist: float = 60.0,
    lens_center_search_px: int = 60,
    lens_center_step_px: int = 5,
):
    """
    Gentle radial lens warp based on bead displacement.

    Returns
    -------
    warped_reg : np.ndarray (float32)
    warped_mask : np.ndarray (bool)
    ok : bool
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.ndimage import map_coordinates

    # ------------------------------------------------------------
    # 1. Extract 2D bead centroids from masks (XY only)
    # ------------------------------------------------------------
    def _centroids_from_mask(mask_zyx):
        zproj = mask_zyx.any(axis=0)
        ys, xs = np.nonzero(zproj)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([ys, xs], axis=1).astype(np.float32)

    fixed_pts = _centroids_from_mask(fixed_mask)
    moving_pts = _centroids_from_mask(moving_mask)

    if fixed_pts.shape[0] < 10 or moving_pts.shape[0] < 10:
        LOG.debug("lens_warp: insufficient beads")
        return None, None, False

    # ------------------------------------------------------------
    # 2. Match beads (nearest neighbor with distance cap)
    # ------------------------------------------------------------
    tree = cKDTree(moving_pts)
    dists, idx = tree.query(fixed_pts, distance_upper_bound=match_max_dist)

    valid = np.isfinite(dists)
    if np.count_nonzero(valid) < 10:
        LOG.debug("lens_warp: insufficient matched beads")
        return None, None, False

    f = fixed_pts[valid]
    m = moving_pts[idx[valid]]

    disp = f - m  # observed displacement field (Y,X)

    # ------------------------------------------------------------
    # 3. Grid search lens center + fit single radial coefficient
    # ------------------------------------------------------------
    h, w = fixed_mask.shape[1:]
    cy0, cx0 = h / 2.0, w / 2.0

    best_err = np.inf
    best_params = None

    for dy in range(-lens_center_search_px, lens_center_search_px + 1, lens_center_step_px):
        for dx in range(-lens_center_search_px, lens_center_search_px + 1, lens_center_step_px):
            cy = cy0 + dy
            cx = cx0 + dx

            r2 = (m[:, 0] - cy) ** 2 + (m[:, 1] - cx) ** 2
            A = np.stack([r2 * (m[:, 0] - cy), r2 * (m[:, 1] - cx)], axis=1)

            # Solve least squares: disp ≈ k * A
            k, _, _, _ = np.linalg.lstsq(A.reshape(-1, 1), disp.reshape(-1), rcond=None)
            k = float(k)

            pred = k * A
            err = np.mean((pred - disp) ** 2)

            if err < best_err:
                best_err = err
                best_params = (cy, cx, k)

    if best_params is None:
        LOG.debug("lens_warp: model fit failed")
        return None, None, False

    cy, cx, k = best_params
    LOG.debug("lens_warp fitted: center=(%.1f,%.1f) k=%.3e mse=%.4g", cy, cx, k, best_err)

    # ------------------------------------------------------------
    # 4. Build warp field
    # ------------------------------------------------------------
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    dy = k * r2 * (yy - cy)
    dx = k * r2 * (xx - cx)

    map_y = yy - dy
    map_x = xx - dx

    # ------------------------------------------------------------
    # 5. Apply warp slice-by-slice
    # ------------------------------------------------------------
    def _warp_volume(vol, order):
        out = np.empty_like(vol, dtype=np.float32)
        if vol.ndim == 3:
            for z in range(vol.shape[0]):
                out[z] = map_coordinates(
                    vol[z].astype(np.float32),
                    [map_y, map_x],
                    order=order,
                    mode="constant",
                    cval=0.0,
                )
        else:  # (Z,C,Y,X)
            for z in range(vol.shape[0]):
                for c in range(vol.shape[1]):
                    out[z, c] = map_coordinates(
                        vol[z, c].astype(np.float32),
                        [map_y, map_x],
                        order=order,
                        mode="constant",
                        cval=0.0,
                    )
        return out

    warped_reg = _warp_volume(moving_reg, order=1)
    warped_mask = _warp_volume(moving_mask.astype(np.float32), order=0) > 0.5

    return warped_reg, warped_mask, True


def _sitk_resample_2d(arr2d: np.ndarray, tx, interp):
    import SimpleITK as sitk
    mov = sitk.GetImageFromArray(arr2d.astype(np.float32, copy=False))
    ref = sitk.GetImageFromArray(np.zeros_like(arr2d, dtype=np.float32))
    res = sitk.Resample(mov, ref, tx, interp, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(res)


def _apply_sitk_tx_slicewise(vol: np.ndarray, tx, interp):
    import SimpleITK as sitk

    if vol.ndim == 3:
        out = np.empty_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            out[z] = _sitk_resample_2d(vol[z], tx, interp)
        return out

    if vol.ndim == 4:
        out = np.empty_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            for c in range(vol.shape[1]):
                out[z, c] = _sitk_resample_2d(vol[z, c], tx, interp)
        return out

    raise ValueError(f"Expected vol.ndim in (3,4), got {vol.ndim}")


def _try_sitk_affine_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, n_iter: int = 200):
    try:
        import SimpleITK as sitk
    except ImportError:
        LOG.warning("SimpleITK not available: skipping affine registration.")
        return None, False

    fixed = sitk.GetImageFromArray(fixed_2d.astype(np.float32, copy=False))
    moving = sitk.GetImageFromArray(moving_2d.astype(np.float32, copy=False))

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
        costFunctionConvergenceFactor=1e7,
    )

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    try:
        out_tx = reg.Execute(fixed, moving)
        return out_tx, True
    except Exception as e:
        LOG.warning("Affine registration failed: %s", str(e))
        return None, False


def _try_sitk_bspline_2d(fixed_2d: np.ndarray, moving_2d: np.ndarray, mesh_size=(2, 2), n_iter: int = 60):
    try:
        import SimpleITK as sitk
    except ImportError:
        LOG.warning("SimpleITK not available: skipping B-spline refinement.")
        return None, False

    fixed = sitk.GetImageFromArray(fixed_2d.astype(np.float32, copy=False))
    moving = sitk.GetImageFromArray(moving_2d.astype(np.float32, copy=False))

    try:
        tx0 = sitk.BSplineTransformInitializer(fixed, mesh_size)
    except Exception as e:
        LOG.warning("B-spline init failed: %s", str(e))
        return None, False

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(tx0, inPlace=False)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=int(n_iter),
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e7,
    )

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    try:
        out_tx = reg.Execute(fixed, moving)
        return out_tx, True
    except Exception as e:
        LOG.warning("B-spline registration failed: %s", str(e))
        return None, False

def _central_err_for_shift(shift_zyx: np.ndarray, box):
    moved = _shift_mask(moving_mask, shift_zyx)
    return _central_dice_error(fixed_mask, moved, box), moved
    