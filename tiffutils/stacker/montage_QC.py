# coding: utf-8
# Author: Miiko Sokka

"""
These functions are used together to create large montages from single-cell tiff stacks.
Will save one or more montages (100 cells per montage) for each channel.

Example usage:

    filenamelist = tiffu.get_filenames(input_path, regex=regex_var, subfolders=False)
    arrays = [tiffu.load_tiff(os.path.join(input_path, f), 4) for f in filenamelist]
    processed_arrays = [
        tiffu.mip(
            tiffu.histogram_stretch(
                tiffu.convert_dtype(a, 'uint8'), [0.1, 99.999]
            )
        ) for a in arrays
    ]
    tiffu.create_QC_montages(processed_arrays, filenamelist, output_path)
    
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def helper_QC_montage(arrays, titles, title, save_as, channel_index):
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


def create_QC_montage(all_arrays, all_titles, output_path, max_per_montage=100):
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
            helper_QC_montage(arrays_chunk, titles_chunk, title, save_as, channel_index=ch)
