# coding: utf-8
# Author: Miiko Sokka

import os

import numpy as np
import tifffile as tiff

from .logging_utils import Timer, get_logger, setup_root_logging
LOG = get_logger(__name__)

def load_tiff(path_to_file, filenames=False, expected_channels=4):
    """
    Load a TIFF stack and ensure it has the expected number of channels.

    Typical use:

    arrays = []
    filenames_loaded = []

    for file in filenames:
        arr, fname = tiffu.load_tiff(os.path.join(input_path, file))
        if arr is not None:
            arrays.append(arr)
            filenames_loaded.append(fname)


    Parameters:
    - path_to_file (str): Full path to the TIFF file, asserting that it is (or supposed to be) of shape ZCYX.
    - expected_channels (int): The expected number of channels in the image.

    Returns:
    - tuple: (np.ndarray or None, str or None)
        - A 4D NumPy array with shape (Z, C, Y, X), or None if loading failed.
        - The filename (basename) of the TIFF file, or None if loading failed.
    
    Behavior:
    - If the input TIFF has shape (ZC, Y, X), it will reshape it to (Z, C, Y, X).
    - If the number of channels is less than `expected_channels`, it pads the array with blank channels (value 65536).
    - Prints the shape changes and final dtype of the array.
    """

    t_total = Timer()
    filename = os.path.basename(path_to_file)

    LOG.debug("start step=load_tiff file=%s filenames=%s expected_channels=%s", filename, filenames, expected_channels)
    
    try:
        array_original = tiff.imread(path_to_file, is_ome=False)
    except Exception:
        LOG.exception(
            "Error loading file=%s. NOTE! A 4D (Z,C,Y,X or ZC,Y,X) TIFF stack is expected."
            "To load generic tiff file, use tifffile.imread(array, is_ome=False)",
            filename,
        )
        return None, None

    array = array_original

    # Handle 3D TIFFs (ZC, Y, X)
    if expected_channels == 4 and len(array.shape) == 3:
        zc, y, x = array.shape
        z = zc // expected_channels
        array = array.reshape(z, expected_channels, y, x)
        LOG.debug("step=load_tiff reshaped ZC->ZCYX file=%s zc=%d z=%d c=%d", filename, zc, z, expected_channels)

    # Add missing channels if needed
    if array.shape[1] < expected_channels:
        missing_channels = expected_channels - array.shape[1]
        # print(f'\tFile {filename} has fewer channels than {expected_channels}. '
        #       f'\tAdding {missing_channels} empty array(s)...')
        z, _, y, x = array.shape
        empty_channels = np.full((z, missing_channels, y, x), 65535, dtype=array.dtype)
        array = np.concatenate((array, empty_channels), axis=1)
        LOG.debug("step=load_tiff padded channels file=%s missing_channels=%d fill_value=%d", filename, missing_channels, 65535)
    
    LOG.debug(
        "done step=load_tiff file=%s shape=%s dtype=%s (original_shape=%s) expected_channels=%d time_s=%.3f",
        filename,
        array.shape,
        array.dtype,
        array_original.shape,
        expected_channels,
        t_total.s(),
    )

    if filenames:
        return array, filename
    else:
        return array