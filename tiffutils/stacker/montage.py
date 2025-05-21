# coding: utf-8
# Author: Miiko Sokka


import numpy as np
import matplotlib.pyplot as plt
import math
import os


def create_3D_montage(
    array, array_x, array_y,
    save_path="montage_rgb_labeled.png",
    cmap_name='inferno',
    spacer=10,
    row_spacer=30,
    label_space=20,
    margin_top=40,
    margin_bottom=40,
    margin_left=40,
    margin_right=40
):
    """
    Create an RGB square montage from a stack of 3D arrays, adding side and bottom projections
    and labeling each panel with its channel number.

    Parameters
    ----------
    array : np.ndarray
        3D array of shape (C, Y, X), where C is the number of channels.
        This array provides the main (top-left) content for each panel.
    array_x : np.ndarray
        3D array of shape (C, Yx, X), providing the bottom projection for each panel.
        It must match the X dimension of `array`.
    array_y : np.ndarray
        3D array of shape (C, Y, Xy), providing the side projection for each panel.
        It must match the Y dimension of `array`.
    save_path : str, optional
        File path to save the resulting montage image (default is "montage_rgb_labeled.png").
    cmap_name : str, optional
        Name of the matplotlib colormap used to convert grayscale to RGB (default is 'inferno').
    spacer : int, optional
        Number of pixels between the inner projections and main image (default is 10).
    row_spacer : int, optional
        Extra spacing in pixels between rows of panels (default is 30).
    label_space : int, optional
        Vertical space above each row to accommodate channel label text (default is 20).
    margin_top, margin_bottom, margin_left, margin_right : int, optional
        Margins around the entire montage in pixels (default is 40 for all).

    Returns
    -------
    None
        The montage image is saved to `save_path`. Nothing is returned.
    """
    
    def normalize(img):

        """
        Helper function
        """
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)


    C, Y, X = array.shape
    _, Yx, _ = array_x.shape
    _, _, Xy = array_y.shape

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

        base_rgb = cmap(normalize(array[i]))[:, :, :3]
        side_rgb = cmap(normalize(array_x[i]))[:, :, :3]
        bottom_rgb = cmap(normalize(array_y[i]))[:, :, :3]

        panel = np.ones((panel_height, panel_width, 3), dtype=np.float32)
        panel[0:Y, 0:X, :] = base_rgb
        panel[0:Y, X + spacer:X + spacer + Xy, :] = bottom_rgb
        panel[Y + spacer:Y + spacer + Yx, 0:X, :] = side_rgb

        montage_rgb[y_start:y_start + panel_height, x_start:x_start + panel_width, :] = panel
        label_positions.append((x_start + 5, y_start - 5, i))  # Above the panel

    dpi = 100
    fig_height = montage_height / dpi
    fig_width = montage_width / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.imshow(montage_rgb, interpolation='none')
    ax.axis("off")

    for x, y, i in label_positions:
        ax.text(
            x, y, f"Channel {i}",
            fontdict={'family': 'Arial', 'size': 22},
            ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1')
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, pad_inches=0.2)
    plt.close()
