# tiffutils/_optional.py
from __future__ import annotations

import importlib
from typing import Any, Optional


class OptionalDependencyError(ImportError):
    pass


def optional_import(module: str, *, extra: Optional[str] = None, conda_name: Optional[str] = None) -> Any:
    """
    Import an optional dependency only when needed.

    Parameters
    ----------
    module : str
        Import name (e.g. "cv2", "matplotlib.pyplot", "SimpleITK").
    extra : str, optional
        Extras group name to suggest, e.g. "opencv", "viz".
    conda_name : str, optional
        Conda package name suggestion for conda-forge.

    Returns
    -------
    Imported module
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        # If it failed on a nested import, let it raise normally
        if e.name != module.split(".")[0]:
            raise

        conda_pkg = conda_name or module.split(".")[0]
        hint = f"conda install -c conda-forge {conda_pkg}"
        if extra:
            hint += f"\n(or: pip install 'tiffutils[{extra}]')"

        raise OptionalDependencyError(
            f"Optional dependency '{module}' is required for this function.\n"
            f"Install it with:\n  {hint}"
        ) from e