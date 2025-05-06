# coding: utf-8
# Author: Miiko Sokka

import os
import numpy as np
import tifffile
from natsort import natsorted
from ..io.filefinder import get_filenames
from ..io.loader import load_tiff

def zcyx_to_tzcyx_subfolders(folder_path, pattern=r'.*\.tiff?$'):
    """
    Load TIFF files from subfolders and stack them into a TZCYX NumPy array.

    Assumes that each subfolder contains ZCYX TIFF files representing Z slices for a timepoint.
    Matching filenames are naturally sorted within each subfolder.

    Parameters:
    - folder_path (str): Path to the main folder containing subfolders.
    - pattern (str): Regex pattern to match TIFF files (default: all .tiff/.tif files).

    Returns:
    - tzcyx (np.ndarray): A NumPy array of shape (T, Z, C, Y, X).
    """
    # subfolders = natsorted([os.path.join(folder_path, d) for d in os.listdir(folder_path)
    #                         if os.path.isdir(os.path.join(folder_path, d))])
    # time_series = []
    # for subfolder in subfolders:
    #     filenames = natsorted(get_filenames(subfolder, regex=pattern, subfolders=False))
    #     z_stack = [tifffile.imread(f) for f in filenames]
    #     zcyx = np.stack(z_stack, axis=0)  # shape: Z, C, Y, X
    #     time_series.append(zcyx)
    # tzcyx = np.stack(time_series, axis=0)  # shape: T, Z, C, Y, X
    # return tzcyx


def tzcyx_to_zcyx_subfolders(tzcyx_array, folder_path, pattern=r'.*\.tiff?$'):
    """
    Split a TZCYX array into ZCYX arrays corresponding to each subfolder (timepoint).

    Parameters:
    - tzcyx_array (np.ndarray): Input array of shape (T, Z, C, Y, X).
    - folder_path (str): Path to the main folder containing subfolders.
    - pattern (str): Regex pattern to match filenames in subfolders.

    Returns:
    - list of np.ndarray: List of arrays, one per subfolder, each with shape (Z, C, Y, X).
    """
    # subfolders = natsorted([os.path.join(folder_path, d) for d in os.listdir(folder_path)
    #                         if os.path.isdir(os.path.join(folder_path, d))])
    # assert len(tzcyx_array) == len(subfolders), "T dimension must match number of subfolders."
    # return [tzcyx_array[t] for t in range(tzcyx_array.shape[0])]


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


def tzcyx_to_zcyx_single_folder(tzcyx_array, folder_path, pattern=r'.*\.tiff?$'):
    """
    Split a TZCYX array into ZCYX arrays for a single folder.

    Parameters:
    - tzcyx_array (np.ndarray): Input array of shape (T, Z, C, Y, X).
    - folder_path (str): Path to the folder containing original TIFF files.
    - pattern (str): Regex pattern to match filenames.

    Returns:
    - list of np.ndarray: List of arrays, one per timepoint, each with shape (Z, C, Y, X).
    """
    # filenames = natsorted(get_filenames(folder_path, regex=pattern, subfolders=False))
    # assert len(tzcyx_array) == len(filenames), "T dimension must match number of files."
    # return [tzcyx_array[t] for t in range(tzcyx_array.shape[0])]