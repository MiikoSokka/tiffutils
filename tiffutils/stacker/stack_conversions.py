# coding: utf-8
# Author: Miiko Sokka

import os
import numpy as np
import tifffile
from natsort import natsorted
from ..io.filefinder import get_filenames
from ..io.loader import load_tiff


def zcyx_to_tzcyx_single_folder(filenamelist, input_path, channelnumber=4):
    """
    Load TIFF files from a single folder and stack them into a TZCYX array.

    Assumes each TIFF file is a ZCYX array representing a timepoint.

    Parameters:
    - folder_path (str): Path to the folder containing TIFF files.
    - pattern (str): Regex pattern to match TIFF files.

    Returns:
    - tzcyx (np.ndarray): A NumPy array of shape (T, Z, C, Y, X).
    """
    
    zcyx_stacks = [load_tiff(os.path.join(input_path, f), expected_channels=channelnumber) for f in filenamelist]
    tzcyx = np.stack(zcyx_stacks, axis=0)
    return tzcyx


def get_YZ_and_ZX_views(array: np.ndarray, XY_pixel_in_nm: float, Z_pixel_in_nm: float):
    """
    Get orthogonal views of a ZCYX (or ZYX) array.

    Reslice a 4D array (Z, C, Y, X) or 3D array (Z, Y, X) into scaled isotropic views,
    and return the scaled arrays.

    Parameters:
    - array (np.ndarray): Input array of shape (Z, C, Y, X) or (Z, Y, X)
    - XY_pixel_in_nm (float): Pixel size in XY dimensions (in nm)
    - Z_pixel_in_nm (float): Pixel size in Z dimension (in nm)

    Returns:
    - array_x_scaled (np.ndarray): Scaled resliced array in (X, C, Y, Z) or (X, Y, Z)
    - array_y_scaled (np.ndarray): Scaled resliced array in (Y, C, Z, X) or (Y, Z, X)
    """
    assert array.ndim in (3, 4), "Array must be 3D (Z, Y, X) or 4D (Z, C, Y, X)"
    scalefactor = Z_pixel_in_nm / XY_pixel_in_nm
    repeat_count = int(round(scalefactor))

    if array.ndim == 4:
        # ZCYX → XCYZ
        array_x = array.transpose(3, 1, 2, 0)
        array_x_scaled = np.repeat(array_x, repeats=repeat_count, axis=3)

        # ZCYX → YCZX
        array_y = array.transpose(2, 1, 0, 3)
        array_y_scaled = np.repeat(array_y, repeats=repeat_count, axis=2)
    else:
        # ZYX → XYZ
        array_x = array.transpose(2, 1, 0)
        array_x_scaled = np.repeat(array_x, repeats=repeat_count, axis=2)

        # ZYX → YZX
        array_y = array.transpose(1, 0, 2)
        array_y_scaled = np.repeat(array_y, repeats=repeat_count, axis=1)

    return array_y_scaled, array_x_scaled


def reorder_channels(hyperstack, order_list):
    """
    Reorders the channels based on a list of new positions.

    Note that the numbers in the list are the original 0-based positions and the position of a number will be the new reordered position.

    When making the list, it is easiest to make two columns, original_order and new_order, in an excel sheet. Number the new_order column based on the new order 
    and then sort the two columns by new_order. Copy the values from original_order column, then transpose and copy the transposed values. Then do echo '<transposed original_order>'|sed 's/\t/,/g'|pbcopy.
    Finally, paste the values into python order_list.
    """

    # Validate that order_list is the same length as the channel dimension
    if len(order_list) != hyperstack.shape[1]:
        raise ValueError(f"order_list length ({len(order_list)}) must match number of channels ({hyperstack.shape[1]})")
    
    return hyperstack[:, order_list, :, :]