# coding: utf-8
# Author: Miiko Sokka

import os
import numpy as np
import tifffile as tiff

def save_tiff(stack, filename, output_folder):
    """
    Save a TIFF image stack to disk with optional datatype conversion.

    Parameters:
    - stack (np.ndarray): The image stack to be saved.
    - filename (str): The name of the output TIFF file.
    - output_folder (str): Path to the directory where the TIFF will be saved.
    - datatype (np.dtype or str, optional): Desired datatype for saving. 
      If None, the original datatype of the input stack is used.

    The function creates the output folder if it does not exist,
    optionally converts the data type, and saves the stack as a TIFF file using the ImageJ format.
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, filename)

    tiff.imwrite(save_path, stack, imagej=True)
    print(f"{filename} saved.")
    print(f"\tShape = {stack.shape}")
    print(f"\tDatatype = {stack.dtype}")