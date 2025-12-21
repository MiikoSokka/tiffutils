# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from ..segmentation.edges import apply_edges, overlay_arrays


def create_3D_montage(
    vol: np.ndarray,
    mask: np.ndarray,
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
    add_labels: bool = True,
    channel_labels: list[str] | None = None,   # NEW
):
    """
    Create an RGB square montage from a 3D or 4D volume, adding side and bottom projections.

    Accepted shapes:
      - 4D: (Z, C, Y, X)
      - 3D: (Z, Y, X)  -> internally treated as (Z, 1, Y, X)

    If add_labels is True:
      - use channel_labels[i] if provided
      - otherwise fall back to "Channel i"
    """

    # --- Normalize shapes to (Z, C, Y, X) ---
    if vol.ndim == 4:
        if mask.ndim != 4 or vol.shape != mask.shape:
            raise ValueError("vol and mask must have the same 4D shape")
        vol_zcyx = vol
        mask_zcyx = mask
        Z, C, Y, X = vol_zcyx.shape

    elif vol.ndim == 3:
        if mask.ndim != 3 or vol.shape != mask.shape:
            raise ValueError("vol and mask must have the same 3D shape")
        Z, Y, X = vol.shape
        C = 1
        vol_zcyx = vol[:, None, :, :]
        mask_zcyx = mask[:, None, :, :]

    else:
        raise ValueError(f"vol must be 3D or 4D, got {vol.shape}")

    # --- Validate channel_labels ---
    if channel_labels is not None:
        if len(channel_labels) != C:
            raise ValueError(
                f"channel_labels length ({len(channel_labels)}) "
                f"must match number of channels ({C})"
            )

    # --- Projections ---
    vol_xy  = np.empty((C, Y, X), dtype=vol_zcyx.dtype)
    vol_xz  = np.empty((C, Z, X), dtype=vol_zcyx.dtype)
    vol_yz  = np.empty((C, Y, Z), dtype=vol_zcyx.dtype)

    mask_xy = np.empty((C, Y, X), dtype=mask_zcyx.dtype)
    mask_xz = np.empty((C, Z, X), dtype=mask_zcyx.dtype)
    mask_yz = np.empty((C, Y, Z), dtype=mask_zcyx.dtype)

    for c in range(C):
        vol_c = vol_zcyx[:, c]
        msk_c = mask_zcyx[:, c]

        vol_xy[c] = np.max(vol_c, axis=0)
        vol_xz[c] = np.max(vol_c, axis=1)
        vol_yz[c] = np.max(vol_c, axis=2).T

        mask_xy[c] = np.max(msk_c, axis=0)
        mask_xz[c] = np.max(msk_c, axis=1)
        mask_yz[c] = np.max(msk_c, axis=2).T

    # --- Edges + overlay ---
    edge_xy = apply_edges(mask_xy)
    edge_xz = apply_edges(mask_xz)
    edge_yz = apply_edges(mask_yz)

    ov_xy = overlay_arrays(vol_xy, edge_xy, alpha=overlay_alpha)
    ov_xz = overlay_arrays(vol_xz, edge_xz, alpha=overlay_alpha)
    ov_yz = overlay_arrays(vol_yz, edge_yz, alpha=overlay_alpha)

    # --- Montage layout ---
    def normalize(img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    _, Yx, _ = ov_xz.shape
    _, _, Xy = ov_yz.shape

    panel_height = Y + spacer + Yx
    panel_width = X + spacer + Xy

    cols = math.ceil(np.sqrt(C))
    rows = math.ceil(C / cols)

    effective_label_space = label_space if add_labels else 0

    montage_height = (
        margin_top
        + effective_label_space
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

        y_start = margin_top + effective_label_space + r * (panel_height + spacer * 2 + row_spacer)
        x_start = margin_left + c * (panel_width + spacer * 2)

        base_rgb   = cmap(normalize(ov_xy[i]))[:, :, :3]
        side_rgb   = cmap(normalize(ov_xz[i]))[:, :, :3]
        bottom_rgb = cmap(normalize(ov_yz[i]))[:, :, :3]

        panel = np.ones((panel_height, panel_width, 3), dtype=np.float32)
        panel[0:Y, 0:X] = base_rgb
        panel[0:Y, X + spacer:X + spacer + Xy] = bottom_rgb
        panel[Y + spacer:Y + spacer + Yx, 0:X] = side_rgb

        montage_rgb[y_start:y_start + panel_height, x_start:x_start + panel_width] = panel

        if add_labels:
            label_positions.append((x_start + 5, y_start - 5, i))

    fig, ax = plt.subplots(
        figsize=(montage_width / 100, montage_height / 100),
        dpi=100,
    )
    ax.imshow(montage_rgb, interpolation='none')
    ax.axis("off")

    if add_labels:
        for x, y, i in label_positions:
            label = (
                channel_labels[i]
                if channel_labels is not None
                else f"Channel {i}"
            )
            ax.text(
                x, y, label,
                fontdict={'family': 'DejaVu Sans', 'size': 22},
                ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none',
                          boxstyle='round,pad=0.1')
            )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out_dir = os.path.dirname(str(save_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(str(save_path), pad_inches=0.2)
    plt.close()