# coding: utf-8
# Author: Miiko Sokka

import numpy as np
from aicssegmentation.core.vessel import filament_3d_wrapper
from skimage.morphology import remove_small_objects

def segmentMetaphaseChromosomes(array_3D, f3_param = [[1, 0.01]], minArea = 4):
    """
    Segment chromosome-like filaments from a 3D fluorescence image using a filament detection algorithm.

    Parameters
    ----------
    array_3D : np.ndarray
        A 3D NumPy array of shape (Z, Y, X), expected to be histogram-stretched and of type uint16.
    f3_param : list of list, optional
        Parameters for the `filament_3d_wrapper` function, controlling filament detection.
        Default is [[1, 0.01]]. This works well for metaphase chromosome spreads
    minArea : int, optional
        Minimum size (in voxels) for an object to be retained in the post-processing step.
        Default is 4.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of the same shape as `array_3D`, of type uint16.
        Values are 0 for background and 65535 for segmented chromosome-like objects.
    """

    # Segmentation
    bw = filament_3d_wrapper(array_3D, f3_param)

    # Post processing
    seg = remove_small_objects(bw>0, min_size=minArea, connectivity=1)
    
    seg_uint16 = (seg * 65535).astype(np.uint16)

    return seg_uint16