"""
Synthetic checks that saved registration transforms reproduce register_3d_stack output.

Run:  python tests/test_registration_transforms.py
"""
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

import tiffutils as tiffu
from tiffutils.processing import registration_centroids as rc


def _beads(shape, n, rng):
    Z, Y, X = shape
    pts = np.column_stack([rng.integers(3, Z - 3, n), rng.integers(30, Y - 30, n), rng.integers(30, X - 30, n)])
    m = np.zeros(shape, bool)
    for z, y, x in pts:
        m[z - 1 : z + 2, y - 2 : y + 3, x - 2 : x + 3] = True
    return m


def _warp_xy(mask_zyx, shift_yx, scale):
    """Deform a mask in-plane (shift + isotropic scale about the centre)."""
    Z, Y, X = mask_zyx.shape
    yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing="ij")
    cy, cx = Y / 2, X / 2
    my = (yy - cy) / scale + cy - shift_yx[0]
    mx = (xx - cx) / scale + cx - shift_yx[1]
    out = np.zeros_like(mask_zyx)
    for z in range(Z):
        out[z] = map_coordinates(mask_zyx[z].astype(np.float32), [my, mx], order=0) > 0.5
    return out


def _make_case(seed, shift, scale):
    rng = np.random.default_rng(seed)
    shape = (10, 256, 256)
    fixed = _beads(shape, 120, rng)
    moving = _warp_xy(fixed, shift, scale)
    # (Z,C,Y,X) moving stack: beads in channel 1, a smooth texture in the others
    zc = rng.random((shape[0], 3, shape[1], shape[2])).astype(np.float32)
    stack = np.empty((shape[0], 3, shape[1], shape[2]), np.uint16)
    stack[:] = (zc * 1000).astype(np.uint16)
    stack[:, 1] = (moving * 4000).astype(np.uint16)
    return fixed, moving, stack


def check_case(seed, shift, scale):
    fixed, moving, stack = _make_case(seed, shift, scale)
    out_plain = tiffu.register_3d_stack(fixed, moving, stack, dice_threshold=0.05, crop=200,
                                        affine_iters=60, smooth_bspline_iters=30)
    out, shv, err, reg = tiffu.register_3d_stack(
        fixed, moving, stack, dice_threshold=0.05, crop=200, affine_iters=60,
        smooth_bspline_iters=30, return_transforms=True)
    assert len(out_plain) == 3
    assert np.array_equal(out, out_plain[0]), "return_transforms changed the output"

    again = tiffu.apply_registration(stack, reg)
    assert again.dtype == out.dtype and np.array_equal(again, out), "apply != fitted output"

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "reg.json"
        tiffu.save_registration(reg, p)
        loaded = tiffu.load_registration(p)
    assert loaded.chain == reg.chain
    again2 = tiffu.apply_registration(stack, loaded)
    assert np.array_equal(again2, out), "apply(loaded) != fitted output"
    print(f"seed={seed} shift={shift} scale={scale}: stages={reg.meta['stages']} err={err:.3f}  OK")
    return reg.meta["stages"]


def check_wrong_shape():
    fixed, moving, stack = _make_case(0, (3, 2), 1.0)
    *_, reg = tiffu.register_3d_stack(fixed, moving, stack, return_transforms=True)
    try:
        tiffu.apply_registration(stack[:, :, :100], reg)
    except ValueError:
        print("shape mismatch raises ValueError  OK")
        return
    raise AssertionError("shape mismatch not detected")


def check_records_roundtrip():
    import SimpleITK as sitk

    aff = sitk.AffineTransform(2)
    aff.SetParameters([1.01, 0.02, -0.01, 0.99, 3.0, -2.0])
    aff.SetFixedParameters([100.0, 120.0])
    bs = sitk.BSplineTransform(2, 3)
    bs.SetTransformDomainOrigin([0.0, 0.0])
    bs.SetTransformDomainPhysicalDimensions([255.0, 255.0])
    bs.SetTransformDomainMeshSize([10, 10])
    bs.SetParameters(np.random.default_rng(1).normal(0, 2, len(bs.GetParameters())).tolist())
    vol = np.random.default_rng(2).random((4, 2, 128, 128)).astype(np.float32)
    chain = [rc._rigid_record(np.array([1.0, 2.5, -3.25], np.float32)), rc._sitk_record(aff),
             rc._lens_record(60.0, 70.0, 3e-9), rc._sitk_record(bs)]
    direct = rc._shift_full_volume(vol, np.array([1.0, 2.5, -3.25], np.float32))
    direct = rc._apply_sitk_tx_slicewise(direct, aff, sitk.sitkLinear)
    direct = rc._apply_lens(direct, 60.0, 70.0, 3e-9, order=1).astype(np.float32)
    direct = rc._apply_sitk_tx_slicewise(direct, bs, sitk.sitkLinear)
    import json
    chain2 = json.loads(json.dumps(chain))
    assert np.array_equal(rc._apply_chain(vol, chain2), direct)
    print("record round trip (rigid+affine+lens+bspline via JSON)  OK")


def check_alt_branch():
    """Force the first affine fit to be bad so the lens>affine>bspline branch runs."""
    import SimpleITK as sitk

    orig = rc._try_sitk_affine_2d
    calls = {"n": 0}

    def flaky(fixed_2d, moving_2d, n_iter=200):
        calls["n"] += 1
        if calls["n"] == 1:
            tx = sitk.AffineTransform(2)
            tx.SetTranslation([45.0, -40.0])
            return tx, True
        return orig(fixed_2d, moving_2d, n_iter=n_iter)

    rc._try_sitk_affine_2d = flaky
    try:
        fixed, moving, stack = _make_case(1, (8, 5), 1.03)
        out, shv, err, reg = tiffu.register_3d_stack(
            fixed, moving, stack, dice_threshold=0.05, crop=200, affine_iters=60,
            smooth_bspline_iters=30, return_transforms=True)
    finally:
        rc._try_sitk_affine_2d = orig
    stages = reg.meta["stages"]
    assert calls["n"] >= 2, "alt branch did not run"
    assert np.array_equal(tiffu.apply_registration(stack, reg), out), "alt chain != fitted output"
    print(f"alt branch: stages={stages} err={err:.3f}  OK")
    return stages


if __name__ == "__main__":
    check_records_roundtrip()
    seen = set()
    for seed, shift, scale in [(0, (6, -4), 1.0), (1, (8, 5), 1.03), (2, (-9, 7), 0.97), (3, (4, 4), 1.06)]:
        seen.update(check_case(seed, shift, scale))
    check_wrong_shape()
    seen.update(check_alt_branch())
    print("stage types exercised end-to-end:", sorted(seen))
