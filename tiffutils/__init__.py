# tiffutils/__init__.py

# --------------------
# Core, always-present imports
# --------------------
from .io.filefinder import (
    get_filenames,
    match_filenames,
    get_project_root,
)
from .io.saver import save_tiff
from .io.loader import load_tiff
from .io.show import show_array

from .processing.resample import resample_z_to_match_xy
from .processing.dtype import convert_dtype
from .processing.modify_histogram import histogram_stretch

from .segmentation.edges import apply_edges, overlay_arrays

from .stacker.projection import (
    mip,
    aip,
    reshape_timepoints_to_channels,
    reshape_channels_to_timepoints,
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
    "apply_edges",
    "overlay_arrays",
    "mip",
    "aip",
    "reshape_timepoints_to_channels",
    "reshape_channels_to_timepoints",
    "create_3D_montage",
]

# --------------------
# Optional: registration (find_centroids, register_3d_stack)
# --------------------
try:
    from .processing.registration_centroids import find_centroids, register_3d_stack
except Exception:
    # registration is optional; don't break `import tiffutils` if it fails
    find_centroids = None
    register_3d_stack = None
else:
    __all__ += ["find_centroids", "register_3d_stack"]

# --------------------
# Optional: segmentation (heavy deps like Cellpose/SAM)
# --------------------
try:
    from .segmentation.segment_nuclei import (
        segment_nuclei_cpsam_3d,
        crop_and_save_nuclei_from_mask,
        stack_single_file,
    )
except Exception:
    segment_nuclei_cpsam_3d = None
    crop_and_save_nuclei_from_mask = None
    stack_single_file = None
else:
    __all__ += [
        "segment_nuclei_cpsam_3d",
        "crop_and_save_nuclei_from_mask",
        "stack_single_file",
    ]

try:
    from .segmentation.segment_features import (
        segmentMetaphaseChromosomes,
        segmentChromosomeTerritories,
    )
except Exception:
    segmentMetaphaseChromosomes = None
    segmentChromosomeTerritories = None
else:
    __all__ += [
        "segmentMetaphaseChromosomes",
        "segmentChromosomeTerritories",
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
