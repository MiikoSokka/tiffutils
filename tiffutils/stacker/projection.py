# coding: utf-8
# Author: Miiko Sokka

import numpy as np

import numpy as np

def mip(array):
    """
    Generate a Maximum Intensity Projection (MIP) over the z-axis for each channel in the input array.

    Parameters:
    - array (np.ndarray): A NumPy array of shape:
        - 3D: (Z, Y, X)
        - 4D: (Z, C, Y, X)
        - 5D: (T, Z, C, Y, X)

    Returns:
    - mip_array (np.ndarray): A NumPy array containing the MIP for each channel:
        - From 3D input: shape (Y, X)
        - From 4D input: shape (C, Y, X)
        - From 5D input: shape (T, C, Y, X)
        - On error: returns the original input array and prints the error.
    """
    try:
        if array.ndim == 3:
            return np.max(array, axis=0)
        elif array.ndim == 4:
            return np.max(array, axis=0)
        elif array.ndim == 5:
            return np.max(array, axis=1)
        else:
            raise ValueError("Input array must be 3D, 4D, or 5D.")
    except Exception as e:
        print(f"[MIP ERROR] {e}")
        return array


import numpy as np

def aip(array):
    """
    Generate an Average Intensity Projection (AIP) over the z-axis for each channel in the input array.

    Parameters:
    - array (np.ndarray): A NumPy array of shape:
        - 3D: (Z, Y, X)
        - 4D: (Z, C, Y, X)
        - 5D: (T, Z, C, Y, X)

    Returns:
    - aip_array (np.ndarray): A NumPy array containing the AIP for each channel:
        - From 3D input: shape (Y, X)
        - From 4D input: shape (C, Y, X)
        - From 5D input: shape (T, C, Y, X)
        - On error: returns the original input array and prints the error.
    """
    try:
        if array.ndim == 3:
            return np.mean(array, axis=0)
        elif array.ndim == 4:
            return np.mean(array, axis=0)
        elif array.ndim == 5:
            return np.mean(array, axis=1)
        else:
            raise ValueError("Input array must be 3D, 4D, or 5D.")
    except Exception as e:
        print(f"[AIP ERROR] {e}")
        return array


def reshape_timepoints_to_channels(array_5D):
    """
    Reshape a 5D array of shape (T, Z, C, Y, X) into a 4D array of shape (Z, T*C, Y, X),
    effectively combining the timepoints and channels into a single dimension.

    Parameters:
    - array_5D (np.ndarray): A 5-dimensional NumPy array with shape (T, Z, C, Y, X)

    Returns:
    - array_4D (np.ndarray): A 4-dimensional NumPy array with shape (Z, T*C, Y, X)
    
    Raises:
    - AssertionError: If the input array is not 5-dimensional
    """
    assert array_5D.ndim == 5, "Input array must be 5-dimensional (T, Z, C, Y, X)"
    t, z, c, y, x = array_5D.shape
    tc = t * c
    array_4D = array_5D.transpose(1, 0, 2, 3, 4).reshape(z, tc, y, x)
    return array_4D


def reshape_channels_to_timepoints(array_4D, t, c):
    """
    Reshape a 4D array of shape (Z, T*C, Y, X) back into a 5D array of shape (T, Z, C, Y, X),
    reversing the transformation done by `reshape_timepoints_to_channels`.

    Parameters:
    - array_4D (np.ndarray): A 4-dimensional NumPy array with shape (Z, T*C, Y, X)
    - t (int): Number of timepoints (T)
    - c (int): Number of channels (C)

    Returns:
    - array_5D (np.ndarray): A 5-dimensional NumPy array with shape (T, Z, C, Y, X)
    
    Raises:
    - AssertionError: If the second dimension of array_4D does not match T*C
    """
    assert array_4D.ndim == 4, "Input array must be 4-dimensional (Z, T*C, Y, X)"
    z, tc, y, x = array_4D.shape
    assert tc == t * c, f"Second dimension ({tc}) does not match T*C ({t}*{c})"
    array_5D = array_4D.reshape(z, t, c, y, x).transpose(1, 0, 2, 3, 4)  # (T, Z, C, Y, X)
    return array_5D

