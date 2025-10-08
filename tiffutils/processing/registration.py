# coding: utf-8
# Author: Miiko Sokka

import os
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation

import numpy as np
from skimage.registration import phase_cross_correlation

def register_arrays(arrays, fiducial_channel):
    """
    Registers a list of image stacks using fiducial beads on the specified channel.

    Parameters:
    - arrays (list of np.ndarray): List of 4D image stacks with shape (Z, C, Y, X)
    - fiducial_channel (int): Index of the channel used for registration

    Returns:
    - registered_arrays (list of np.ndarray): List with the fixed array first, followed by registered arrays
    """
    if not arrays:
        raise ValueError("Input array list is empty.")

    fixed_image = arrays[0]
    registered_arrays = [fixed_image]

    for j, moving_image in enumerate(arrays[1:], start=1):
        try:
            print(f"\nRegistering: Timepoint {j}")

            # Compute 3D shift on fiducial channel
            shift, error, diffphase = phase_cross_correlation(
                fixed_image[:, fiducial_channel, :, :],
                moving_image[:, fiducial_channel, :, :],
                upsample_factor=1
            )
            print("ZYX Shift:", shift)

            # Apply shift to all channels
            shift_z, shift_y, shift_x = map(int, shift)
            registered_image = np.zeros_like(moving_image)
            for c in range(moving_image.shape[1]):
                registered_image[:, c, :, :] = np.roll(
                    moving_image[:, c, :, :],
                    shift=(shift_z, shift_y, shift_x),
                    axis=(0, 1, 2)
                )

            registered_arrays.append(registered_image)

        except Exception as e:
            print(f"Error registering moving image at timepoint {j}: {e}")
            registered_arrays.append(moving_image)  # Append unmodified image if registration fails

    print("Registration completed.")
    return registered_arrays
