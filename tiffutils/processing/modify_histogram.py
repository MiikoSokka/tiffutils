# coding: utf-8
# Author: Miiko Sokka

import numpy as np

def histogram_stretch(input_array, intensity_scaling_param=[1, 99], verbose=False):
    """
    Apply histogram stretching using percentile clipping to a 2D, 3D, 4D, or 5D NumPy array.

    Parameters:
    -----------
    input_array : np.ndarray
        A 2D (Y, X), 3D (Z, Y, X), 4D (Z, C, Y, X), or 5D (T, Z, C, Y, X) NumPy array.
    intensity_scaling_param : list or tuple of two floats
        The lower and upper percentiles used for contrast stretching, e.g., [1, 99].

    Returns:
    --------
    np.ndarray
        The stretched array with the same shape and dtype as the input. The values are clipped
        and scaled to enhance contrast but remain in the original data type's range.
    
    Notes:
    ------
    - For 2D and 3D arrays, stretching is applied globally.
    - For 4D arrays, stretching is applied independently for each channel.
    - For 5D arrays, stretching is applied independently for each timepoint and channel.
    - Stretching does not normalize to [0, 1].
    - For [0, 1] normalization, use tiffutils.processing.dtype(array, dtype=float32)
    """

    p_lower, p_upper = intensity_scaling_param

    if p_lower < 0 or p_upper > 100:
        raise ValueError("\tPercentile values must be between 0 and 100.")

    arr_dtype = input_array.dtype

    if input_array.ndim == 2:
        print(f'\tHistogram stretching a 2D array of shape {input_array.shape}')
        # 2D: Y, X
        p1 = np.percentile(input_array, p_lower)
        p99 = np.percentile(input_array, p_upper)
    
        if verbose:
            print(f"\tp1={p1}, p99={p99}")

        if p99 - p1 < 1e-5:
            print("\tWarning: Percentile range too small. Skipping normalization.")
            return input_array.copy()

        clipped = np.clip(input_array, p1, p99)
        stretched = ((clipped - p1) / (p99 - p1)) * (np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0)
        return stretched.astype(arr_dtype)

    elif input_array.ndim == 3:
        print(f'Histogram stretching a 3D array of shape {input_array.shape}')
        # 3D: Z, Y, X
        p1 = np.percentile(input_array, p_lower)
        p99 = np.percentile(input_array, p_upper)

        if verbose:
            print(f"\tp1={p1}, p99={p99}")

        if p99 - p1 < 1e-5:
            print("\tWarning: Percentile range too small. Skipping normalization.")
            return input_array.copy()

        clipped = np.clip(input_array, p1, p99)
        stretched = ((clipped - p1) / (p99 - p1)) * (np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0)
        return stretched.astype(arr_dtype)

    elif input_array.ndim == 4:
        print(f'Histogram stretching a 4D array of shape {input_array.shape}')
        # 4D: Z, C, Y, X
        Z, C, Y, X = input_array.shape
        stretched = np.empty_like(input_array)

        for c in range(C):
            channel = input_array[:, c, :, :]
            p1 = np.percentile(channel, p_lower)
            p99 = np.percentile(channel, p_upper)

            if verbose:
                print(f"\tChannel {c}: p1={p1}, p99={p99}")

            if p99 - p1 < 1e-5:
                print(f"\tWarning: Channel {c} percentile range too small. Skipping normalization.")
                stretched[:, c, :, :] = channel
            else:
                clipped = np.clip(channel, p1, p99)
                scaled = ((clipped - p1) / (p99 - p1)) * (
                    np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0
                )
                stretched[:, c, :, :] = scaled.astype(arr_dtype)

        return stretched

    elif input_array.ndim == 5:
        print(f'Histogram stretching a 5D array of shape {input_array.shape}')
        # 5D: T, Z, C, Y, X
        T, Z, C, Y, X = input_array.shape
        stretched = np.empty_like(input_array)

        for t in range(T):
            for c in range(C):
                subarray = input_array[t, :, c, :, :]
                p1 = np.percentile(subarray, p_lower)
                p99 = np.percentile(subarray, p_upper)

                if verbose:
                    print(f"\tTime {t}, Channel {c}: p1={p1}, p99={p99}")

                if p99 - p1 < 1e-5:
                    print(f"\tWarning: T{t} C{c} percentile range too small. Skipping normalization.")
                    stretched[t, :, c, :, :] = subarray
                else:
                    clipped = np.clip(subarray, p1, p99)
                    scaled = ((clipped - p1) / (p99 - p1)) * (
                        np.iinfo(arr_dtype).max if np.issubdtype(arr_dtype, np.integer) else 1.0
                    )
                    stretched[t, :, c, :, :] = scaled.astype(arr_dtype)

        return stretched

    else:
        print(f"\tUnsupported array with shape {input_array.shape}. Skipping.")
        return input_array.copy()
