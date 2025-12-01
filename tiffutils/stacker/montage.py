# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from ..segmentation.edges import apply_edges, overlay_arrays

def create_3D_montage(
    vol_zcyx: np.ndarray,
    mask_zcyx: np.ndarray,
    save_path="montage.png",
    cmap_name='inferno',
    spacer=10,
    row_spacer=30,
    label_space=20,
    margin_top=40,
    margin_bottom=40,
    margin_left=40,
    margin_right=40,
    overlay_alpha: float = 0.5,
):
    """
    Create an RGB square montage from a 4D ZCYX array, adding side and bottom projections
    and labeling each panel with its channel number.

    For each projection (XY, XZ, YZ):
      - compute the projection of vol_zcyx and mask_zcyx
      - detect edges in the mask projections with tiffu.apply_edges
      - overlay edges onto the vol projections with tiffu.overlay_arrays
      - use the resulting overlays to build the montage.
    """

    # --- Basic checks ---
    if vol_zcyx.ndim != 4:
        raise ValueError(f"vol_zcyx must have shape (Z, C, Y, X), got {vol_zcyx.shape}")
    if mask_zcyx.ndim != 4:
        raise ValueError(f"mask_zcyx must have shape (Z, C, Y, X), got {mask_zcyx.shape}")
    if vol_zcyx.shape != mask_zcyx.shape:
        raise ValueError(
            f"vol_zcyx and mask_zcyx must have the same shape, got "
            f"{vol_zcyx.shape} and {mask_zcyx.shape}"
        )

    Z, C, Y, X = vol_zcyx.shape

    # --- Build projections per channel (intensity + mask), but DO NOT overlay yet ---
    # vol_xy   : (C, Y, X)
    # vol_xz   : (C, Z, X)
    # vol_yz   : (C, Y, Z)
    # mask_xy  : (C, Y, X)
    # mask_xz  : (C, Z, X)
    # mask_yz  : (C, Y, Z)
    vol_xy  = np.empty((C, Y, X), dtype=vol_zcyx.dtype)
    vol_xz  = np.empty((C, Z, X), dtype=vol_zcyx.dtype)
    vol_yz  = np.empty((C, Y, Z), dtype=vol_zcyx.dtype)

    mask_xy = np.empty((C, Y, X), dtype=mask_zcyx.dtype)
    mask_xz = np.empty((C, Z, X), dtype=mask_zcyx.dtype)
    mask_yz = np.empty((C, Y, Z), dtype=mask_zcyx.dtype)

    for c in range(C):
        vol_c = vol_zcyx[:, c, :, :]   # (Z, Y, X)
        msk_c = mask_zcyx[:, c, :, :]  # (Z, Y, X)

        # --- Intensity projections ---
        # XY: max over Z -> (Y, X)
        vol_xy[c] = np.max(vol_c, axis=0)
        # XZ: max over Y -> (Z, X)
        vol_xz[c] = np.max(vol_c, axis=1)
        # YZ: max over X -> (Z, Y) then transpose -> (Y, Z)
        vol_yz[c] = np.max(vol_c, axis=2).T

        # --- Mask projections ---
        mask_xy[c] = np.max(msk_c, axis=0)
        mask_xz[c] = np.max(msk_c, axis=1)
        mask_yz[c] = np.max(msk_c, axis=2).T

    # --- Edge detection on the projected mask stacks (3D) ---
    edge_xy = apply_edges(mask_xy)  # (C, Y, X)
    edge_xz = apply_edges(mask_xz)  # (C, Z, X)
    edge_yz = apply_edges(mask_yz)  # (C, Y, Z)

    # --- Overlay edges onto intensity projections, ONCE per projection type ---
    # overlay_arrays works on 3D (C, Y, X)-like arrays, so shapes match.
    ov_xy = overlay_arrays(vol_xy, edge_xy, alpha=overlay_alpha)  # (C, Y, X)
    ov_xz = overlay_arrays(vol_xz, edge_xz, alpha=overlay_alpha)  # (C, Z, X)
    ov_yz = overlay_arrays(vol_yz, edge_yz, alpha=overlay_alpha)  # (C, Y, Z)

    # -----------------------------------------------------------
    # The rest is essentially your original montage function,
    # but using ov_xy / ov_xz / ov_yz instead of raw projections.
    # -----------------------------------------------------------
    def normalize(img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    _, Yx, _ = ov_xz.shape   # Yx = Z  (height of XZ projection)
    _, _, Xy = ov_yz.shape   # Xy = Z  (width of YZ projection)

    panel_height = Y + spacer + Yx
    panel_width = X + spacer + Xy

    cols = math.ceil(np.sqrt(C))
    rows = math.ceil(C / cols)

    montage_height = (
        margin_top
        + label_space
        + rows * panel_height
        + (rows - 1) * (spacer * 2 + row_spacer)
        + margin_bottom
    )

    montage_width = (
        margin_left
        + cols * panel_width
        + (cols - 1) * spacer * 2
        + margin_right
    )

    montage_rgb = np.ones((montage_height, montage_width, 3), dtype=np.float32)
    cmap = plt.get_cmap(cmap_name)

    label_positions = []

    for i in range(C):
        r = i // cols
        c = i % cols

        y_start = margin_top + label_space + r * (panel_height + spacer * 2 + row_spacer)
        x_start = margin_left + c * (panel_width + spacer * 2)

        # Map overlaid intensity+edges (uint8) -> [0,1] -> colormap RGB
        base_rgb   = cmap(normalize(ov_xy[i]))[:, :, :3]   # (Y,  X, 3)
        side_rgb   = cmap(normalize(ov_xz[i]))[:, :, :3]   # (Yx, X, 3)
        bottom_rgb = cmap(normalize(ov_yz[i]))[:, :, :3]   # (Y,  Xy,3)

        panel = np.ones((panel_height, panel_width, 3), dtype=np.float32)
        panel[0:Y, 0:X, :] = base_rgb
        panel[0:Y, X + spacer:X + spacer + Xy, :] = bottom_rgb
        panel[Y + spacer:Y + spacer + Yx, 0:X, :] = side_rgb

        montage_rgb[y_start:y_start + panel_height, x_start:x_start + panel_width, :] = panel
        label_positions.append((x_start + 5, y_start - 5, i))

    dpi = 100
    fig_height = montage_height / dpi
    fig_width = montage_width / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(montage_rgb, interpolation='none')
    ax.axis("off")

    for x, y, i in label_positions:
        ax.text(
            x, y, f"Channel {i}",
            fontdict={'family': 'DejaVu Sans', 'size': 22},
            ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1')
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out_dir = os.path.dirname(str(save_path))
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(str(save_path), pad_inches=0.2)
    plt.close()