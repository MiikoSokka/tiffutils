# tiffutils/__init__.py
from __future__ import annotations

from .io.logging_utils import (
    DEFAULT_FMT,
    DEFAULT_DATEFMT,
    get_logger,
    level_from_verbose,
    setup_root_logging,
    ContextLogger,
    Timer,
)

from ._api import (
    get_filenames,
    match_filenames,
    get_project_root,
    save_tiff,
    load_tiff,
    show_array,
    resample_z_to_match_xy,
    convert_dtype,
    histogram_stretch,
    apply_edges,
    overlay_arrays,
    mip,
    aip,
    reshape_timepoints_to_channels,
    reshape_channels_to_timepoints,
    create_3D_montage,
    # optional wrappers (safe because they're wrappers)
    find_centroids,
    register_3d_stack,
    segment_nuclei_cpsam_3d,
    crop_and_save_nuclei_from_mask,
    stack_single_file,
    segmentMetaphaseChromosomes,
    segmentChromosomeTerritories,
    segmentSpeckles
)

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
    "apply_edges",
    "overlay_arrays",
    "mip",
    "aip",
    "reshape_timepoints_to_channels",
    "reshape_channels_to_timepoints",
    "create_3D_montage",
    "find_centroids",
    "register_3d_stack",
    "segment_nuclei_cpsam_3d",
    "crop_and_save_nuclei_from_mask",
    "stack_single_file",
    "segmentMetaphaseChromosomes",
    "segmentChromosomeTerritories",
    "segmentSpeckles",
    "DEFAULT_FMT",
    "DEFAULT_DATEFMT",
    "get_logger",
    "level_from_verbose",
    "setup_root_logging",
    "ContextLogger",
    "Timer",
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
