# tiffutils/processing/__init__.py

# Always-safe imports (no heavy dependencies)
from .dtype import convert_dtype
from .modify_histogram import histogram_stretch
from .resample import resample_z_to_match_xy

# Optional helper: make stubs for optional functions
def _missing_dependency_stub(msg: str):
    def _stub(*args, **kwargs):
        raise ImportError(msg)
    return _stub

# registration.py
try:
    from .registration import register_arrays, register_stacks_from_paths
except ImportError as e:
    _err = str(e)
    register_arrays = _missing_dependency_stub(
        "register_arrays() is unavailable because "
        "tiffutils.processing.registration could not be imported. "
        f"Original error: {_err}"
    )
    register_stacks_from_paths = _missing_dependency_stub(
        "register_stacks_from_paths() is unavailable because "
        "tiffutils.processing.registration could not be imported. "
        f"Original error: {_err}"
    )

# registration_blob.py
try:
    from .registration_blob import register_and_save_batch
except ImportError as e:
    _err = str(e)
    register_and_save_batch = _missing_dependency_stub(
        "register_and_save_batch() is unavailable because "
        "tiffutils.processing.registration_blob could not be imported. "
        f"Original error: {_err}"
    )

# registration_centroids.py — this is where aicssegmentation comes in
try:
    from .registration_centroids import find_centroids, register_3d_stack
except ImportError as e:
    _err = str(e)
    find_centroids = _missing_dependency_stub(
        "find_centroids() is unavailable because "
        "tiffutils.processing.registration_centroids or one of its "
        "optional dependencies (likely `aicssegmentation`) could not "
        f"be imported. Original error: {_err}"
    )
    register_3d_stack = _missing_dependency_stub(
        "register_3d_stack() is unavailable because "
        "tiffutils.processing.registration_centroids or one of its "
        "optional dependencies (likely `aicssegmentation`) could not "
        f"be imported. Original error: {_err}"
    )

__all__ = [
    "convert_dtype",
    "histogram_stretch",
    "resample_z_to_match_xy",
    "register_arrays",
    "register_stacks_from_paths",
    "register_and_save_batch",
    "find_centroids",
    "register_3d_stack",
]