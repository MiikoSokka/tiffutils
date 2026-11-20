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

import json
import os
from dataclasses import dataclass, field

import numpy as np
import tiffutils as tiffu
from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)


def centroids_from_mask_3d(
    mask_zyx: np.ndarray,
    *,
    sphere_diameter: int | None = 6,
    connectivity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Take a 3D *binary* mask and return:
      - centroids_mask: boolean 3D mask with a small sphere drawn at each centroid
      - centroids_3d: (N, 3) float array of (z, y, x) centroids

    Parameters
    ----------
    mask_zyx : np.ndarray
        3D mask (Z, Y, X). Nonzero voxels are treated as foreground.
    sphere_diameter : int | None, optional
        Diameter (voxels) of centroid spheres. If None or <=1, only the centroid voxels are True.
    connectivity : int, optional
        Connectivity for connected-component labeling. (1 is usually appropriate for 3D.)

    Returns
    -------
    centroids_mask : np.ndarray (bool)
        3D boolean mask (Z, Y, X) where True marks voxels inside centroid spheres.
    centroids_3d : np.ndarray (float)
        Array of centroids, shape (N, 3) with (z, y, x) coordinates.
    """
    if mask_zyx.ndim != 3:
        raise ValueError(f"mask_zyx must be 3D (Z, Y, X). Got shape {mask_zyx.shape}")

    bw = mask_zyx.astype(bool, copy=False)

    if not np.any(bw):
        return np.zeros_like(bw, dtype=bool), np.zeros((0, 3), dtype=float)

    lbl = label(bw, connectivity=connectivity)
    props = regionprops(lbl)

    if not props:
        return np.zeros_like(bw, dtype=bool), np.zeros((0, 3), dtype=float)

    centroids_3d = np.asarray([p.centroid for p in props], dtype=float)

    # place points at nearest voxel to each centroid
    points = np.zeros_like(bw, dtype=bool)
    zyx = np.rint(centroids_3d).astype(int)

    Z, Y, X = bw.shape
    for z, y, x in zyx:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            points[z, y, x] = True

    if sphere_diameter is None or sphere_diameter <= 1:
        return points, centroids_3d

    radius = max(1, int(sphere_diameter) // 2)
    centroids_mask = binary_dilation(points, footprint=ball(radius))

    return centroids_mask, centroids_3d


def find_centroids(
    array_3D,
    s3_param=[[1, 0.08]],
    minArea=20,
    sphere_diameter=6,
    gaussian_smoothing_sigma=1,
    return_labels=False,
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
    return_labels : bool, optional
        If True, also return the labeled segmentation of the spots (see below). Default False.

    Returns
    -------
    centroids_mask : np.ndarray (bool)
        3D boolean mask (Z, Y, X) where True marks voxels inside centroid spheres.
    centroids_3d : np.ndarray
        Array of centroids, shape (N, 3) with (z, y, x) coordinates (float).
    labeled_mask : np.ndarray (int), only if ``return_labels=True``
        3D label image (Z, Y, X) of the segmented spots (after size filter and watershed split).
        Label ``i + 1`` is the spot whose centroid is ``centroids_3d[i]``; 0 is background.
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
        if return_labels:
            return centroids_mask, centroids_3d, np.zeros(mask.shape, dtype=np.int32)
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
        if return_labels:
            return centroids_mask, centroids_3d, labeled_mask
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

    if return_labels:
        return centroids_mask, centroids_3d, labeled_mask
    return centroids_mask, centroids_3d


#######################################
# 3D registration function with helpers
#######################################

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
    smooth_bspline_mesh_size: tuple[int, int] = (8, 8),
    smooth_bspline_iters: int = 120,
    return_transforms: bool = False,
    ):

    """
    Updated workflow (coarse->fine, ALWAYS-APPLY refinement):
      Stage 0: full-volume rigid (dz via Z-profiles + XY PCC on 2D projections) -> global Dice (logged)
      Stage 1/1b: NCC rigid (absolute, from original position), selecting BEST by CENTRAL Dice -> commit rigid baseline
      Stage 2: 2D affine (SimpleITK) on 2D projections; apply slice-by-slice (ALWAYS applied if it runs)
      Stage 3: lens warp (optional; if available, ALWAYS applied if it runs)  [order: affine > lens]
      Stage 4: 2D B-spline residual (SimpleITK) on 2D projections; apply slice-by-slice (ALWAYS applied if it runs)

    Returns
    -------
    registered : np.ndarray
    shift_vec : np.ndarray  (dz,dy,dx) rigid shift used as baseline before affine/lens/bspline
    error : float           FINAL FULL-volume Dice error (1-dice)
    registration : Registration, only if ``return_transforms=True``
        The accepted transform chain (rigid shift, then any accepted affine / lens /
        B-spline stages, in application order) plus metadata. Re-apply it to a stack of
        the same shape with ``apply_registration`` (bit-identical to ``registered``), or
        persist it with ``save_registration``.
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

    # Accepted transforms, in application order. Appended to in lockstep with every
    # accepted update of ``current_reg`` below, so that ``_apply_chain`` on the original
    # stack reproduces the returned registered stack exactly.
    chain: list[dict] = []
    reg_params = {
        "dice_threshold": float(dice_threshold),
        "crop": int(crop),
        "affine_iters": int(affine_iters),
        "smooth_bspline_mesh_size": [int(v) for v in smooth_bspline_mesh_size],
        "smooth_bspline_iters": int(smooth_bspline_iters),
    }

    def _result(out_arr, shift_vec, err):
        if not return_transforms:
            return out_arr, shift_vec, float(err)
        reg = Registration(
            chain=[dict(rec) for rec in chain],
            meta=_registration_meta(moving_original, shift_vec, err, reg_params, chain),
        )
        return out_arr, shift_vec, float(err), reg

    fixed_2d = fixed_mask.any(axis=0).astype(np.float32)
    moving_2d = moving_mask.any(axis=0).astype(np.float32)

    # -------------------------
    # Helpers local to this function
    # -------------------------
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

    def _central_err_for_shift(shift_zyx: np.ndarray, box):
        moved = _shift_mask(moving_mask, shift_zyx)
        return _central_dice_error(fixed_mask, moved, box), moved

    def _maybe_accept(stage_name: str, cand_reg: np.ndarray, cand_mask: np.ndarray, cand_err: float):
        nonlocal current_reg, current_mask, current_err
        if float(cand_err) <= float(current_err):
            current_reg = cand_reg
            current_mask = cand_mask
            current_err = float(cand_err)
            LOG.debug("%s accepted global_err=%.4f", stage_name, float(current_err))
            return True
        else:
            LOG.debug("%s rejected cand_err=%.4f > prev_err=%.4f", stage_name, float(cand_err), float(current_err))
            return False

    # -------------------------
    # Stage 0 — Full-volume rigid (PCC) (logged; may early-stop)
    # -------------------------
    dz0 = _estimate_dz_from_z_profiles(fixed_mask, moving_mask)
    dy0, dx0, pcc_score = _pcc_shift_2d(fixed_2d, moving_2d)

    shift0 = np.asarray([float(dz0), float(dy0), float(dx0)], dtype=np.float32)
    moved0 = _shift_mask(moving_mask, shift0)
    err0 = _dice_error_from_masks(fixed_mask, moved0)

    LOG.debug(
        "stage0 dz=%.2f dy=%.2f dx=%.2f pcc_score=%s global_err=%.4f thr=%.4f",
        float(dz0),
        float(dy0),
        float(dx0),
        str(pcc_score),
        float(err0),
        float(dice_threshold),
    )

    # If stage0 is already good, return immediately (optional shortcut)
    if float(err0) <= float(dice_threshold):
        reg0 = _shift_full_volume(moving_original, shift0)
        out0 = _cast_like_input(reg0.astype(np.float32, copy=False), orig_dtype)
        LOG.info("done stage=stage0_pcc_full error=%.4f (stop: global<=thr)", float(err0))
        chain.append(_rigid_record(shift0))
        return _result(out0, shift0.copy(), float(err0))

    # -------------------------
    # Stage 1 / 1b — NCC: select BEST shift by CENTRAL Dice (absolute, from original moving)
    # -------------------------
    crop_box0 = _crop_box_xy(int(crop))

    best_ncc = {
        "shift": None,
        "central_err": float("inf"),
        "ncc_score": float("-inf"),
        "box": crop_box0,
    }

    # Stage 1 (single crop)
    dy1, dx1, ncc_score1 = _xy_shift_from_ncc_template_absolute(
        fixed_2d=fixed_2d,
        moving_mask_zyx_unshifted=moving_mask,
        crop_box=crop_box0,
    )
    shift1 = np.asarray([float(dz0), float(dy1), float(dx1)], dtype=np.float32)
    central_err1, _ = _central_err_for_shift(shift1, crop_box0)

    LOG.debug(
        "stage1 (ABS, central-only) crop=%s ncc_score=%.4f shift=(%.2f,%.2f,%.2f) central_err=%.4f",
        str(crop_box0),
        float(ncc_score1),
        float(shift1[0]),
        float(shift1[1]),
        float(shift1[2]),
        float(central_err1),
    )

    if float(central_err1) < float(best_ncc["central_err"]):
        best_ncc.update(
            {"shift": shift1, "central_err": float(central_err1), "ncc_score": float(ncc_score1), "box": crop_box0}
        )

    proceed_to_refine = float(central_err1) <= float(dice_threshold)

    # Stage 1b (diminishing crops)
    if not proceed_to_refine:
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
            central_err_k, _ = _central_err_for_shift(shift_k, box)

            LOG.debug(
                "stage1b (ABS, central-only) iter=%d crop_size=%d box=%s ncc_score=%.4f shift=(%.2f,%.2f,%.2f) central_err=%.4f best_central_err=%.4f",
                k + 1,
                cur_size,
                str(box),
                float(ncc_score),
                float(shift_k[0]),
                float(shift_k[1]),
                float(shift_k[2]),
                float(central_err_k),
                float(best_ncc["central_err"]),
            )

            if float(central_err_k) < float(best_ncc["central_err"]):
                best_ncc.update(
                    {"shift": shift_k, "central_err": float(central_err_k), "ncc_score": float(ncc_score), "box": box}
                )

            if float(central_err_k) <= float(dice_threshold):
                proceed_to_refine = True
                break

        if not proceed_to_refine:
            LOG.debug("stage1b ended without central<=thr; proceeding anyway with best central shift")

    if best_ncc["shift"] is None:
        LOG.warning("stage1/1b: no NCC result; falling back to stage0 shift")
        rigid_shift = shift0.copy()
    else:
        rigid_shift = best_ncc["shift"].copy()

    # Commit rigid baseline (this is what you return as shift_vec)
    current_shift = rigid_shift.copy()
    current_reg = _shift_full_volume(moving_original, current_shift)
    current_mask = _shift_mask(moving_mask, current_shift)

    current_err = _dice_error_from_masks(fixed_mask, current_mask)
    chain.append(_rigid_record(current_shift))
    LOG.debug(
        "rigid_baseline stage=stage1_best_ncc_by_central box=%s ncc_score=%.4f central_err=%.4f global_err=%.4f shift=(%.2f,%.2f,%.2f)",
        str(best_ncc.get("box", crop_box0)),
        float(best_ncc.get("ncc_score", float("-inf"))),
        float(best_ncc.get("central_err", float("inf"))),
        float(current_err),
        float(current_shift[0]),
        float(current_shift[1]),
        float(current_shift[2]),
    )

    # -------------------------
    # Stage 2 — Affine
    # -------------------------

    alt_refine_taken = False  # <- used to skip the standard Stage 3/4 below if we run the alt branch

    # snapshot state BEFORE affine, so we can try the alternate ordering from the same baseline
    pre_aff_reg = current_reg
    pre_aff_mask = current_mask
    pre_aff_err = float(current_err)

    fixed_aff_2d = fixed_mask.any(axis=0).astype(np.float32)
    moving_aff_2d = current_mask.any(axis=0).astype(np.float32)

    tx_aff, ok_aff = _try_sitk_affine_2d(fixed_aff_2d, moving_aff_2d, n_iter=int(affine_iters))
    if ok_aff:
        import SimpleITK as sitk

        cand_reg = _apply_sitk_tx_slicewise(current_reg, tx_aff, sitk.sitkLinear)

        cand_mask = np.zeros_like(current_mask, dtype=bool)
        for z in range(current_mask.shape[0]):
            cand_mask[z] = (
                _sitk_resample_2d(current_mask[z].astype(np.float32), tx_aff, sitk.sitkNearestNeighbor) > 0.5
            )

        cand_err = _dice_error_from_masks(fixed_mask, cand_mask)
        aff_accepted = _maybe_accept("stage2_affine", cand_reg, cand_mask, cand_err)
        if aff_accepted:
            chain.append(_sitk_record(tx_aff))

        # ------------------------------------------------------------
        # Alternate branch only if affine was REJECTED by global Dice:
        #   lens -> affine -> bspline  (starting from PRE-affine baseline)
        # ------------------------------------------------------------
        if not aff_accepted:
            LOG.debug("stage2_affine rejected; trying alt branch lens>affine>bspline from pre-affine baseline")

            # restore baseline for alternate attempt
            alt_reg = pre_aff_reg
            alt_mask = pre_aff_mask
            alt_err = pre_aff_err
            alt_chain = list(chain)  # rigid only (affine was rejected)

            # (A) Lens first
            lens_reg, lens_mask, ok_lens, lens_params = _try_lens_warp(
                fixed_mask=fixed_mask,
                moving_reg=alt_reg,
                moving_mask=alt_mask,
                match_max_dist=150.0,
                lens_center_search_px=150,
                lens_center_step_px=3,
            )
            if ok_lens:
                alt_chain.append(_lens_record(*lens_params))
                alt_reg = lens_reg.astype(np.float32, copy=False)
                alt_mask = lens_mask.astype(bool, copy=False)
                alt_err = float(_dice_error_from_masks(fixed_mask, alt_mask))
                LOG.debug("alt_stage3_lens candidate global_err=%.4f (baseline=%.4f)", alt_err, pre_aff_err)
            else:
                LOG.debug("alt_stage3_lens ok=0 (skipped/failed)")
                alt_reg, alt_mask, alt_err = pre_aff_reg, pre_aff_mask, pre_aff_err

            # (B) Affine after lens (only if lens ran successfully)
            if ok_lens:
                fixed_aff2_2d = fixed_mask.any(axis=0).astype(np.float32)
                moving_aff2_2d = alt_mask.any(axis=0).astype(np.float32)

                tx_aff2, ok_aff2 = _try_sitk_affine_2d(
                    fixed_aff2_2d, moving_aff2_2d, n_iter=int(affine_iters)
                )
                if ok_aff2:
                    cand_reg2 = _apply_sitk_tx_slicewise(alt_reg, tx_aff2, sitk.sitkLinear)

                    cand_mask2 = np.zeros_like(alt_mask, dtype=bool)
                    for z in range(alt_mask.shape[0]):
                        cand_mask2[z] = (
                            _sitk_resample_2d(alt_mask[z].astype(np.float32), tx_aff2, sitk.sitkNearestNeighbor) > 0.5
                        )

                    cand_err2 = float(_dice_error_from_masks(fixed_mask, cand_mask2))
                    LOG.debug("alt_stage2_affine candidate global_err=%.4f (after lens=%.4f)", cand_err2, alt_err)
                    alt_chain.append(_sitk_record(tx_aff2))
                    alt_reg, alt_mask, alt_err = cand_reg2, cand_mask2, cand_err2
                else:
                    LOG.debug("alt_stage2_affine ok=0 (skipped/failed)")

            # (C) B-spline residual after lens+affine
            if ok_lens:
                fixed_nr2_2d = fixed_mask.any(axis=0).astype(np.float32)
                moving_nr2_2d = alt_mask.any(axis=0).astype(np.float32)

                tx_nr2, ok_nr2 = _try_sitk_bspline_2d(
                    fixed_nr2_2d,
                    moving_nr2_2d,
                    mesh_size=tuple(map(int, smooth_bspline_mesh_size)),
                    n_iter=int(smooth_bspline_iters),
                )
                if ok_nr2:
                    cand_reg3 = _apply_sitk_tx_slicewise(alt_reg, tx_nr2, sitk.sitkLinear)

                    cand_mask3 = np.zeros_like(alt_mask, dtype=bool)
                    for z in range(alt_mask.shape[0]):
                        cand_mask3[z] = (
                            _sitk_resample_2d(alt_mask[z].astype(np.float32), tx_nr2, sitk.sitkNearestNeighbor) > 0.5
                        )

                    cand_err3 = float(_dice_error_from_masks(fixed_mask, cand_mask3))
                    LOG.debug("alt_stage4_bspline candidate global_err=%.4f", cand_err3)
                    alt_chain.append(_sitk_record(tx_nr2))
                    alt_reg, alt_mask, alt_err = cand_reg3, cand_mask3, cand_err3
                else:
                    LOG.debug("alt_stage4_bspline ok=0 (skipped/failed)")

            # Final accept of alternate branch (global Dice only)
            if float(alt_err) <= float(pre_aff_err):
                current_reg = alt_reg
                current_mask = alt_mask
                current_err = float(alt_err)
                chain = alt_chain
                alt_refine_taken = True
                LOG.debug("alt_branch accepted global_err=%.4f (baseline=%.4f)", float(current_err), pre_aff_err)
            else:
                LOG.debug("alt_branch rejected alt_err=%.4f > baseline=%.4f", float(alt_err), pre_aff_err)

    else:
        LOG.debug("stage2_affine ok=0 (skipped/failed)")

    if not alt_refine_taken:
        # -------------------------
        # Stage 3 — Lens warp
        # -------------------------
        lens_reg, lens_mask, ok_lens, lens_params = _try_lens_warp(
            fixed_mask=fixed_mask,
            moving_reg=current_reg,
            moving_mask=current_mask,
            match_max_dist=150.0,
            lens_center_search_px=150,
            lens_center_step_px=3,
        )
        if ok_lens:
            cand_reg = lens_reg.astype(np.float32, copy=False)
            cand_mask = lens_mask.astype(bool, copy=False)
            cand_err = _dice_error_from_masks(fixed_mask, cand_mask)
            if _maybe_accept("stage3_lens", cand_reg, cand_mask, cand_err):
                chain.append(_lens_record(*lens_params))
        else:
            LOG.debug("stage3_lens ok=0 (skipped/failed)")

    if not alt_refine_taken:
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

            cand_reg = _apply_sitk_tx_slicewise(current_reg, tx_nr, sitk.sitkLinear)

            cand_mask = np.zeros_like(current_mask, dtype=bool)
            for z in range(current_mask.shape[0]):
                cand_mask[z] = (
                    _sitk_resample_2d(current_mask[z].astype(np.float32), tx_nr, sitk.sitkNearestNeighbor) > 0.5
                )

            cand_err = _dice_error_from_masks(fixed_mask, cand_mask)
            if _maybe_accept("stage4_bspline", cand_reg, cand_mask, cand_err):
                chain.append(_sitk_record(tx_nr))
        else:
            LOG.debug("stage4_bspline ok=0 (skipped/failed)")

    # -------------------------
    # Finalize + return (ALWAYS)
    # -------------------------
    out = _cast_like_input(current_reg.astype(np.float32, copy=False), orig_dtype)
    LOG.info(
        "done stage=final rigid_shift=(%.2f,%.2f,%.2f) error=%.4f",
        float(current_shift[0]),
        float(current_shift[1]),
        float(current_shift[2]),
        float(current_err),
        )
    return _result(out, current_shift.copy(), float(current_err))


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

    Returns (dy, dx, score) where (dy,dx) is the translation to apply to moving so that
    the moving crop aligns to fixed (in the coordinate convention used by nd_shift).
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
    params : (cy, cx, k) or None
        The fitted lens model; re-apply with ``_apply_lens``.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import map_coordinates

    def _centroids_from_mask(mask_zyx):
        zproj = mask_zyx.any(axis=0)
        ys, xs = np.nonzero(zproj)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([ys, xs], axis=1).astype(np.float32)

    fixed_pts = _centroids_from_mask(fixed_mask)
    moving_pts = _centroids_from_mask(moving_mask)

    if fixed_pts.shape[0] < 10 or moving_pts.shape[0] < 10:
        LOG.debug("lens_warp: insufficient beads fixed=%d moving=%d", int(fixed_pts.shape[0]), int(moving_pts.shape[0]))
        return None, None, False, None

    tree = cKDTree(moving_pts)
    dists, idx = tree.query(fixed_pts, distance_upper_bound=float(match_max_dist))

    valid = np.isfinite(dists)
    if np.count_nonzero(valid) < 10:
        LOG.debug("lens_warp: insufficient matched beads (<=%d within %.1f px)", int(np.count_nonzero(valid)), float(match_max_dist))
        return None, None, False, None

    f = fixed_pts[valid]
    m = moving_pts[idx[valid]]
    disp = (f - m).astype(np.float32)  # (Y,X)

    h, w = fixed_mask.shape[1:]
    cy0, cx0 = h / 2.0, w / 2.0

    best_err = np.inf
    best_params = None

    lens_center_search_px = int(lens_center_search_px)
    lens_center_step_px = int(max(1, lens_center_step_px))

    for dcy in range(-lens_center_search_px, lens_center_search_px + 1, lens_center_step_px):
        for dcx in range(-lens_center_search_px, lens_center_search_px + 1, lens_center_step_px):
            cy = cy0 + float(dcy)
            cx = cx0 + float(dcx)

            r2 = (m[:, 0] - cy) ** 2 + (m[:, 1] - cx) ** 2
            A = np.stack([r2 * (m[:, 0] - cy), r2 * (m[:, 1] - cx)], axis=1).astype(np.float32)

            # disp ≈ k * A  (single scalar)
            k, _, _, _ = np.linalg.lstsq(A.reshape(-1, 1), disp.reshape(-1), rcond=None)
            k = float(k)

            pred = k * A
            err = float(np.mean((pred - disp) ** 2))

            if err < best_err:
                best_err = err
                best_params = (cy, cx, k)

    if best_params is None:
        LOG.debug("lens_warp: model fit failed")
        return None, None, False, None

    cy, cx, k = best_params
    LOG.debug("lens_warp fitted: center=(%.1f,%.1f) k=%.3e mse=%.4g matches=%d", cy, cx, k, best_err, int(np.count_nonzero(valid)))

    warped_reg = _apply_lens(moving_reg, cy, cx, k, order=1)
    warped_mask = _apply_lens(moving_mask.astype(np.float32, copy=False), cy, cx, k, order=0) > 0.5

    return (
        warped_reg.astype(np.float32, copy=False),
        warped_mask.astype(bool, copy=False),
        True,
        (float(cy), float(cx), float(k)),
    )


def _apply_lens(vol: np.ndarray, cy: float, cx: float, k: float, order: int) -> np.ndarray:
    """
    Apply the radial lens model (centre cy,cx; strength k) slice-by-slice to a (Z,Y,X) or
    (Z,C,Y,X) volume. Output is float32.
    """
    from scipy.ndimage import map_coordinates

    h, w = vol.shape[-2:]

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

    out = np.empty_like(vol, dtype=np.float32)
    if vol.ndim == 3:
        for z in range(vol.shape[0]):
            out[z] = map_coordinates(
                vol[z].astype(np.float32, copy=False),
                [map_y, map_x],
                order=int(order),
                mode="constant",
                cval=0.0,
            )
    else:  # (Z,C,Y,X)
        for z in range(vol.shape[0]):
            for c in range(vol.shape[1]):
                out[z, c] = map_coordinates(
                    vol[z, c].astype(np.float32, copy=False),
                    [map_y, map_x],
                    order=int(order),
                    mode="constant",
                    cval=0.0,
                )
    return out


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

# -----------------------------------------------------------------------------
# Transform records, re-application and persistence
#
# ``register_3d_stack(..., return_transforms=True)`` returns a ``Registration``: the
# accepted stages as plain JSON-able records, in application order. The same
# ``_apply_chain`` code path is used for re-applying them, so a saved registration
# reproduces the originally registered stack exactly.
# -----------------------------------------------------------------------------

REGISTRATION_SCHEMA = 1

_SITK_RECORD_TYPES = {
    "AffineTransform": "sitk_affine",
    "BSplineTransform": "sitk_bspline",
}


def _tiffutils_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("tiffutils"))
    except Exception:
        return "unknown"


def _rigid_record(shift_zyx) -> dict:
    return {
        "type": "rigid",
        "shift_zyx": [float(v) for v in np.asarray(shift_zyx, dtype=np.float32)],
    }


def _lens_record(cy: float, cx: float, k: float) -> dict:
    return {"type": "lens", "center_yx": [float(cy), float(cx)], "k": float(k)}


def _sitk_record(tx) -> dict:
    import SimpleITK as sitk

    name = tx.GetName()
    if name == "CompositeTransform":
        # reg.Execute() wraps its result in a composite; keep the nesting and order.
        return {
            "type": "sitk_composite",
            "dimension": int(tx.GetDimension()),
            "transforms": [_sitk_record(tx.GetNthTransform(i)) for i in range(tx.GetNumberOfTransforms())],
        }
    if name not in _SITK_RECORD_TYPES:
        raise ValueError(f"Cannot record SimpleITK transform of type {name!r}")
    rec = {
        "type": _SITK_RECORD_TYPES[name],
        "dimension": int(tx.GetDimension()),
        "parameters": [float(v) for v in tx.GetParameters()],
        "fixed_parameters": [float(v) for v in tx.GetFixedParameters()],
    }
    if name == "BSplineTransform":
        if not hasattr(tx, "GetOrder"):
            tx = sitk.BSplineTransform(tx)  # downcast a generic Transform
        rec["order"] = int(tx.GetOrder()) if hasattr(tx, "GetOrder") else 3
    return rec


def _record_to_sitk(rec: dict):
    import SimpleITK as sitk

    dim = int(rec["dimension"])
    if rec["type"] == "sitk_composite":
        comp = sitk.CompositeTransform(dim)
        for sub in rec["transforms"]:
            comp.AddTransform(_record_to_sitk(sub))
        return comp
    if rec["type"] == "sitk_affine":
        tx = sitk.AffineTransform(dim)
    elif rec["type"] == "sitk_bspline":
        tx = sitk.BSplineTransform(dim, int(rec.get("order", 3)))
    else:
        raise ValueError(f"Not a SimpleITK record: {rec.get('type')!r}")
    # Fixed parameters (grid/centre definition) must be set before the parameters.
    tx.SetFixedParameters([float(v) for v in rec["fixed_parameters"]])
    tx.SetParameters([float(v) for v in rec["parameters"]])
    return tx


def _apply_chain(vol: np.ndarray, chain: list[dict]) -> np.ndarray:
    """
    Apply a recorded transform chain to a (Z,Y,X) or (Z,C,Y,X) volume.
    Returns float32 (cast back to the input dtype is done by the caller).
    """
    import SimpleITK as sitk

    out = vol
    for rec in chain:
        kind = rec["type"]
        if kind == "rigid":
            out = _shift_full_volume(out, np.asarray(rec["shift_zyx"], dtype=np.float32))
        elif kind in ("sitk_affine", "sitk_bspline", "sitk_composite"):
            out = _apply_sitk_tx_slicewise(out, _record_to_sitk(rec), sitk.sitkLinear)
        elif kind == "lens":
            cy, cx = rec["center_yx"]
            out = _apply_lens(out, cy, cx, rec["k"], order=1).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown transform record type: {kind!r}")
    return out


def _registration_meta(moving_original, shift_vec, err, reg_params, chain) -> dict:
    return {
        "tiffutils_version": _tiffutils_version(),
        "input_shape": [int(v) for v in moving_original.shape],
        "input_dtype": str(moving_original.dtype),
        "dice_error": float(err),
        "rigid_shift_zyx": [float(v) for v in np.asarray(shift_vec, dtype=np.float32)],
        "stages": [rec["type"] for rec in chain],
        "params": dict(reg_params),
    }


@dataclass
class Registration:
    """Accepted registration transforms (application order) plus metadata."""

    chain: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"schema": REGISTRATION_SCHEMA, "chain": self.chain, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Registration":
        schema = d.get("schema")
        if schema != REGISTRATION_SCHEMA:
            raise ValueError(
                f"Unsupported registration schema {schema!r} (this tiffutils reads {REGISTRATION_SCHEMA})"
            )
        return cls(chain=[dict(r) for r in d["chain"]], meta=dict(d.get("meta", {})))


def apply_registration(moving_original: np.ndarray, registration) -> np.ndarray:
    """
    Re-apply a registration (``Registration`` or its ``to_dict()``) to a moving stack
    ``(Z,C,Y,X)`` or ``(Z,Y,X)``. For the stack the registration was fitted on, the result
    is identical to the ``registered`` array returned by ``register_3d_stack``.
    """
    reg = registration if isinstance(registration, Registration) else Registration.from_dict(registration)

    recorded = reg.meta.get("input_shape")
    if recorded is not None and list(moving_original.shape) != list(recorded):
        raise ValueError(
            f"Registration was fitted on shape {tuple(recorded)} but is applied to {tuple(moving_original.shape)}"
        )

    out = _apply_chain(moving_original, reg.chain)
    return _cast_like_input(out.astype(np.float32, copy=False), moving_original.dtype)


def registration_from_dict(d: dict) -> Registration:
    return Registration.from_dict(d)


def save_registration(registration: Registration, path) -> None:
    """Write a registration as JSON (atomically: temp file, then rename)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(registration.to_dict(), f, indent=1)
    os.replace(tmp, path)


def load_registration(path) -> Registration:
    with open(path) as f:
        return Registration.from_dict(json.load(f))
