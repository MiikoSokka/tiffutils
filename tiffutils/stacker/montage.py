# coding: utf-8
# Author: Miiko Sokka

import os
import matplotlib.pyplot as plt
import numpy as np

def create_montage(arrays, titles, title, save_as, channel_index):
    """
    Creates and saves a montage of 3D arrays using a specific channel slice.

    Parameters:
    - arrays (list of np.ndarray): List of 3D arrays (C, Y, X).
    - titles (list of str): List of titles for each subplot.
    - title (str): Title for the entire figure.
    - save_as (str): Full path to save the montage image.
    - channel_index (int): Index of the channel to use (C-axis).
    """
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    fig.suptitle(title, fontsize=20)

    for i, ax in enumerate(axes.ravel()):
        if i < len(arrays):
            try:
                if arrays[i].ndim != 3:
                    raise ValueError(f"Array dimension is {arrays[i].ndim}, expected 3D.")
                ax.imshow(arrays[i][channel_index, :, :], cmap='gray')
            except Exception as e:
                print(f"[ERROR] Skipping array at index {i} with title '{titles[i]}': {e}")
                ax.text(0.5, 0.5, titles[i], fontsize=8, ha='center', va='center')
            ax.set_title(titles[i], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as:
        plt.savefig(save_as, dpi=300)
    plt.close(fig)

def save_all_montages(all_arrays, all_titles, output_path, max_per_montage=100):
    """
    Generates montages from a list of image arrays for each channel.

    Parameters:
    - all_arrays (list of np.ndarray): List of image arrays (C, Y, X).
    - all_titles (list of str): Titles for each array.
    - output_path (str): Path to save montage images.
    - max_per_montage (int): Maximum number of images per montage.
    """
    os.makedirs(output_path, exist_ok=True)

    # Determine number of channels based on the first valid array
    num_channels = 0
    for arr in all_arrays:
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            num_channels = arr.shape[0]
            break

    if num_channels == 0:
        print("[ERROR] No valid 3D arrays found to determine channel count.")
        return

    for ch in range(num_channels):
        print(f"Creating montages for channel {ch}")
        num_montages = (len(all_arrays) + max_per_montage - 1) // max_per_montage
        for i in range(num_montages):
            start = i * max_per_montage
            end = min((i + 1) * max_per_montage, len(all_arrays))
            arrays_chunk = all_arrays[start:end]
            titles_chunk = all_titles[start:end]
            title = f"Montage {i+1} - Channel {ch}"
            save_as = os.path.join(output_path, f"montage_{i+1}_ch{ch}.png")
            create_montage(arrays_chunk, titles_chunk, title, save_as, channel_index=ch)
