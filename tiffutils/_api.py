# tiffutils/_api.py
from __future__ import annotations

from typing import Any


# ---- core wrappers (light) ----
def get_filenames(*args: Any, **kwargs: Any):
    from .io.filefinder import get_filenames as f
    return f(*args, **kwargs)

def match_filenames(*args: Any, **kwargs: Any):
    from .io.filefinder import match_filenames as f
    return f(*args, **kwargs)

def get_project_root(*args: Any, **kwargs: Any):
    from .io.filefinder import get_project_root as f
    return f(*args, **kwargs)

def save_tiff(*args: Any, **kwargs: Any):
    from .io.saver import save_tiff as f
    return f(*args, **kwargs)

def load_tiff(*args: Any, **kwargs: Any):
    from .io.loader import load_tiff as f
    return f(*args, **kwargs)

def resample_z_to_match_xy(*args: Any, **kwargs: Any):
    from .processing.resample import resample_z_to_match_xy as f
    return f(*args, **kwargs)

def convert_dtype(*args: Any, **kwargs: Any):
    from .processing.dtype import convert_dtype as f
    return f(*args, **kwargs)

def histogram_stretch(*args: Any, **kwargs: Any):
    from .processing.modify_histogram import histogram_stretch as f
    return f(*args, **kwargs)

def mip(*args: Any, **kwargs: Any):
    from .stacker.projection import mip as f
    return f(*args, **kwargs)

def aip(*args: Any, **kwargs: Any):
    from .stacker.projection import aip as f
    return f(*args, **kwargs)

def reshape_timepoints_to_channels(*args: Any, **kwargs: Any):
    from .stacker.projection import reshape_timepoints_to_channels as f
    return f(*args, **kwargs)

def reshape_channels_to_timepoints(*args: Any, **kwargs: Any):
    from .stacker.projection import reshape_channels_to_timepoints as f
    return f(*args, **kwargs)

def create_3D_montage(*args: Any, **kwargs: Any):
    from .stacker.montage import create_3D_montage as f
    return f(*args, **kwargs)


# ---- OPTIONAL / heavy wrappers ----
def show_array(*args: Any, **kwargs: Any):
    # likely requires matplotlib
    from .io.show import show_array as f
    return f(*args, **kwargs)

def apply_edges(*args: Any, **kwargs: Any):
    from .segmentation.edges import apply_edges as f
    return f(*args, **kwargs)

def overlay_arrays(*args: Any, **kwargs: Any):
    from .segmentation.edges import overlay_arrays as f
    return f(*args, **kwargs)

def find_centroids(*args: Any, **kwargs: Any):
    from .processing.registration_centroids import find_centroids as f
    return f(*args, **kwargs)

def register_3d_stack(*args: Any, **kwargs: Any):
    from .processing.registration_centroids import register_3d_stack as f
    return f(*args, **kwargs)

def segment_nuclei_cpsam_3d(*args: Any, **kwargs: Any):
    from .segmentation.segment_nuclei import segment_nuclei_cpsam_3d as f
    return f(*args, **kwargs)

def crop_and_save_nuclei_from_mask(*args: Any, **kwargs: Any):
    from .segmentation.segment_nuclei import crop_and_save_nuclei_from_mask as f
    return f(*args, **kwargs)

def stack_single_file(*args: Any, **kwargs: Any):
    from .segmentation.segment_nuclei import stack_single_file as f
    return f(*args, **kwargs)

def segmentMetaphaseChromosomes(*args: Any, **kwargs: Any):
    from .segmentation.segment_features import segmentMetaphaseChromosomes as f
    return f(*args, **kwargs)

def segmentChromosomeTerritories(*args: Any, **kwargs: Any):
    from .segmentation.segment_features import segmentChromosomeTerritories as f
    return f(*args, **kwargs)

def segmentSpeckles(*args: Any, **kwargs: Any):
    from .segmentation.segment_features import segmentSpeckles as f
    return f(*args, **kwargs)
