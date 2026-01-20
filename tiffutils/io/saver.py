# coding: utf-8
# Author: Miiko Sokka

import os

import numpy as np
import tifffile as tiff

from .logging_utils import Timer, get_logger, setup_root_logging

LOG = get_logger(__name__)

def save_tiff(array, save_path):
    """
    Save a TIFF image array to disk with optional datatype conversion.

    Parameters:
    - array (np.ndarray): The image array to be saved.
    - filename (str): The name of the output TIFF file.
    - output_folder (str): Path to the directory where the TIFF will be saved.
    - datatype (np.dtype or str, optional): Desired datatype for saving. 
      If None, the original datatype of the input array is used.

    The function creates the output folder if it does not exist,
    optionally converts the data type, and saves the array as a TIFF file using the ImageJ format.
    """
    LOG.debug("start step=save_tiff")
    t_total = Timer()

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tiff.imwrite(save_path, array, imagej=True)

    LOG.debug(
        "done step=save_tiff file=%s dir=%s shape=%s dtype=%s time_s=%.3f",
        os.path.basename(save_path),
        os.path.dirname(save_path),
        getattr(array, "shape", None),
        getattr(array, "dtype", None),
        t_total.s(),
    )