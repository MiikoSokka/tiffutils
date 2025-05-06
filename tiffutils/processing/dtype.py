# coding: utf-8
# Author: Miiko Sokka

import numpy as np

def convert_dtype(array: np.ndarray, dtype: str) -> np.ndarray:
    """
    Convert a NumPy array to the specified data type with appropriate value scaling.

    Parameters:
    ----------
    array : np.ndarray
        Input NumPy array with any numeric type.
    dtype : str
        Target data type. Must be one of: 'uint8', 'uint16', or 'float32'.

    Returns:
    -------
    np.ndarray
        Converted NumPy array with the specified data type.

    Notes:
    -----
    - When converting to 'float32', the array is scaled to the 0-1 range based on the
      minimum and maximum values of the input array.
    - When converting from float or larger integer types to 'uint8' or 'uint16',
      values are clipped and scaled to the target type's range.
    """
    dtype = dtype.lower()
    if dtype not in ('uint8', 'uint16', 'float32'):
        raise ValueError("dtype must be one of: 'uint8', 'uint16', 'float32'")

    if dtype == 'float32':
        array_min = array.min()
        array_max = array.max()
        if array_max == array_min:
            return np.zeros_like(array, dtype=np.float32)
        array = (array - array_min) / (array_max - array_min)
        return array.astype(np.float32)

    elif dtype == 'uint8':
        # Scale to 0-255
        array = np.clip(array, array.min(), array.max())
        array = (array - array.min()) / (array.max() - array.min()) * 255
        return array.astype(np.uint8)

    elif dtype == 'uint16':
        # Scale to 0-65535
        array = np.clip(array, array.min(), array.max())
        array = (array - array.min()) / (array.max() - array.min()) * 65535
        return array.astype(np.uint16)