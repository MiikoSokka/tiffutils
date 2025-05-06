# coding: utf-8
# Author: Miiko Sokka

import os
import numpy as np
import tifffile as tiff

def load_tiff(path_to_file, expected_channels=4):
    """
    Load a TIFF stack and ensure it has the expected number of channels.

    Parameters:
    - path_to_file (str): Full path to the TIFF file, asserting that it is (or supposed to be) of shape ZCYX.
    - expected_channels (int): The expected number of channels in the image.

    Returns:
    - np.ndarray: A 4D NumPy array with shape (Z, C, Y, X). Channels are expanded if necessary.
    
    Behavior:
    - If the input TIFF has shape (ZC, Y, X), it will reshape it to (Z, C, Y, X).
    - If the number of channels is less than `expected_channels`, it pads the array with blank channels (value 65536).
    - Prints the shape changes and final dtype of the array.
    """
    
    try:
        img = tiff.imread(path_to_file, is_ome=False)
    except Exception as e:
        print(f"Error loading {os.path.basename(path_to_file)}: {e}. NOTE! A 4D (Z,C,Y,X or ZC,Y,X) TIFF stack is expected.")
        return None

    print(f"Loading {os.path.basename(path_to_file)} with shape {img.shape} and dtype {img.dtype}")

    # Handle 3D TIFFs (ZC, Y, X)
    if expected_channels == 4 and len(img.shape) == 3:
        zc, y, x = img.shape
        print('\tReshaping', os.path.basename(path_to_file), '\t', img.shape)
        z = zc // expected_channels
        img = img.reshape(z, expected_channels, y, x)
        print('\tShape after reshaping:', img.shape)

    # Add missing channels if needed
    if img.shape[1] < expected_channels:
        missing_channels = expected_channels - img.shape[1]
        print(f'\tFile {os.path.basename(path_to_file)} has fewer channels than {expected_channels}. '
              f'\tAdding {missing_channels} empty array(s)...')
        z, _, y, x = img.shape
        empty_channels = np.full((z, missing_channels, y, x), 65536, dtype=img.dtype)
        img = np.concatenate((img, empty_channels), axis=1)

    return img