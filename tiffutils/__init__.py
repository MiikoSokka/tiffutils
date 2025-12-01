# tiffutils/__init__.py

from .io.filefinder import (
    get_filenames,
    match_filenames,
    get_project_root,
)
from .io.saver import save_tiff
from .io.loader import load_tiff
from .io.show import show_array

from .processing.resample import resample_z_to_match_xy

from .processing.dtype import convert_dtype  # adjust based on actual file names
from .processing.modify_histogram import histogram_stretch

from .segmentation.crop_nuclei import (
    segment_nuclei_cpsam_3d,
    crop_and_save_nuclei_from_mask,
    stack_single_file,
)

from .segmentation.segment_features import segmentMetaphaseChromosomes, segmentChromosomeTerritories

from .segmentation.edges import apply_edges, overlay_arrays

from .stacker.projection import (
    mip,
    aip,
    reshape_timepoints_to_channels,
    reshape_channels_to_timepoints
    )

from .stacker.montage import create_3D_montage

__all__ = [
    "get_filenames",
    "match_filenames",
    "get_project_root",
    "save_tiff",
    "load_tiff",
    "show_array",
    "resample_z_to_match_xy",
    "convert_dtype",
    "histogram_stretch",
    "segment_nuclei_cpsam_3d",
    "crop_and_save_nuclei_from_mask",
    "stack_single_file",
    "segmentMetaphaseChromosomes",
    "segmentChromosomeTerritories",
    "apply_edges", "overlay_arrays",
    "mip",
    "aip",
    "reshape_timepoints_to_channels",
    "reshape_channels_to_timepoints",
    "create_3D_montage"
]



##### Old init:

# from . import io
# from . import processing
# from . import stacker
# from . import segmentation

# from .io.filefinder import get_filenames
# from .io.filefinder import match_filenames
# from .io.filefinder import get_project_root
# from .io.saver import save_tiff
# from .io.loader import load_tiff

# from .processing.dtype import convert_dtype  # adjust based on actual file names
# from .processing.modify_histogram import histogram_stretch
# from .processing.registration import register_arrays, register_stacks_from_paths
# from .processing.registration_blob import register_and_save_batch

# from .processing.registration_centroids import find_centroids, register_3d_stack

# from .stacker.stack_conversions import (
#     zcyx_to_tzcyx_single_folder,
#     get_YZ_and_ZX_views,
#     reorder_channels
#     )
# from .stacker.montage_QC import create_QC_montage

# __all__ = [
#     "get_filenames", "match_filenames", "get_project_root",
#     "save_tiff",
#     "load_tiff",
#     "convert_dtype",
#     "histogram_stretch",
#     "register_arrays",
#     "register_stacks_from_paths",
#     "register_and_save_batch",
#     "find_centroids",
#     "register_3d_stack",
#     "zcyx_to_tzcyx_single_folder",
#     "get_YZ_and_ZX_views",
#     "reorder_channels",
#     "mip",
#     "aip",
#     "reshape_timepoints_to_channels",
#     "reshape_channels_to_timepoints",
#     "io", "processing", "stacker",
#     "create_QC_montage",

#     "create_3D_montage",
#     "segmentMetaphaseChromosomes"
# ]
