# coding: utf-8
# Author: Miiko Sokka

# from .._optional import optional_import
import numpy as np
import matplotlib.pyplot as plt

def show_array(*arrays):
    """
    Display 1–4 2D NumPy arrays side by side.

    Parameters
    ----------
    *arrays : np.ndarray
        One to four 2D arrays of shape (Y, X)
    """
    if not 1 <= len(arrays) <= 4:
        raise ValueError(f"Expected 1–4 arrays, got {len(arrays)}")

    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(
                f"Array {i} must be a 2D NumPy array, got shape={getattr(arr, 'shape', None)}"
            )

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, arr in zip(axes[0], arrays):
        im = ax.imshow(
            arr,
            cmap="inferno",
            interpolation="none",
            vmin=arr.min(),
            vmax=arr.max(),
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout()
    plt.show()