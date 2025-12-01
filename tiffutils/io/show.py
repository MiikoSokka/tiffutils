# coding: utf-8
# Author: Miiko Sokka

import numpy as np
import matplotlib.pyplot as plt

def show_array(arr, title=None):
    """
    Display a 2D NumPy array (Y, X) in a Jupyter Notebook.

    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (Y, X)
    title : str, optional
        Optional title for the plot
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (Y,X). Got shape: {arr.shape}")

    plt.figure(figsize=(4,4))
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()