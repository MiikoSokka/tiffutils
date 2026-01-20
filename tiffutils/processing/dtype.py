# coding: utf-8
# Author: Miiko Sokka

import numpy as np

from ..io.logging_utils import get_logger, Timer

LOG = get_logger(__name__)

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
    t = Timer()
    dtype = dtype.lower()

    LOG.debug(
        "start step=convert_dtype shape=%s src_dtype=%s target=%s min=%s max=%s",
        getattr(array, "shape", None),
        getattr(array, "dtype", None),
        dtype,
        float(np.min(array)) if array.size else None,
        float(np.max(array)) if array.size else None,
    )

    if dtype not in ('uint8', 'uint16', 'float32'):
        raise ValueError("dtype must be one of: 'uint8', 'uint16', 'float32'")

    if dtype == 'float32':
        array_min = array.min()
        array_max = array.max()
        if array_max == array_min:

            LOG.warning(
                "step=convert_dtype constant_input array_min==array_max==%s -> returning zeros (float32)",
                float(array_min),
            )
            return np.zeros_like(array, dtype=np.float32)
        array = (array - array_min) / (array_max - array_min)
        out = array.astype(np.float32)

        LOG.debug("done step=convert_dtype target=float32 time_s=%.3f", t.s())
        return out

    elif dtype == 'uint8':
        # Scale to 0-255
        amin = array.min()
        amax = array.max()
        if amax == amin:
            LOG.warning(
                "step=convert_dtype constant_input array_min==array_max==%s -> returning zeros (uint8)",
                float(amin),
            )
            return np.zeros_like(array, dtype=np.uint8)
        array = np.clip(array, amin, amax)
        array = (array - amin) / (amax - amin) * 255
        out = array.astype(np.uint8)
        LOG.debug("done step=convert_dtype target=uint8 time_s=%.3f", t.s())
        return out

    elif dtype == 'uint16':
        # Scale to 0-65535
        amin = array.min()
        amax = array.max()
        if amax == amin:
            LOG.warning(
                "step=convert_dtype constant_input array_min==array_max==%s -> returning zeros (uint16)",
                float(amin),
            )
            return np.zeros_like(array, dtype=np.uint16)
        array = np.clip(array, amin, amax)
        array = (array - amin) / (amax - amin) * 65535
        out = array.astype(np.uint16)
        LOG.debug("done step=convert_dtype target=uint16 time_s=%.3f", t.s())
        return out