# coding: utf-8
# Author: Miiko Sokka

import os
import re
import subprocess
from typing import List, Tuple, Union

from natsort import natsorted

from .logging_utils import get_logger, Timer, setup_root_logging

LOG = get_logger(__name__)

def get_filenames(input_path, regex=r'.*\.tiff?$', subfolders=False):
    """
    Retrieve a naturally sorted list of filenames matching a given regex pattern from a directory.

    Parameters:
    - input_path (str): Path to the main directory.
    - regex (str): Regular expression to match filenames. Default is '.*\\.tiff?$'.
    - subfolders (bool) = False (default): Search only in the main folder. If True, require subdirectories and validate single matching file across them.
    - subfolders (bool) = True: Require subdirectories and validate single matching file across them. Return file list and directory list

    Returns:
    - list[str]: A naturally sorted list of filenames found (only file names, not full paths).

    Raises:
    - FileNotFoundError: If the input_path does not exist.
    - ValueError: If subfolders=True and no subdirectories are found.
    """

    t = Timer()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Directory not found: {input_path}")

    if not subfolders:
        filenamelist = [
            file for file in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, file)) and
               re.search(regex, file) and
               not file.startswith('._')
        ]
        out = natsorted(filenamelist)
        LOG.debug(
            "done step=get_filenames path=%s subfolders=%s regex=%s n_files=%d time_s=%.3f",
            input_path,
            subfolders,
            regex,
            len(out),
            t.s(),
        )
        return out
    
    # subfolders=True: Check all subdirectories and validate files
    dirs_paths = natsorted([
        os.path.join(input_path, d) for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ])

    if not dirs_paths:
        raise ValueError("No subdirectories found in main directory.")

    all_files = [
        f for f in os.listdir(dirs_paths[0])
        if os.path.isfile(os.path.join(dirs_paths[0], f)) and
           re.search(regex, f) and
           not f.startswith('._')
    ]

    dirs_paths_bases = [os.path.basename(path) for path in dirs_paths]

    files_out = natsorted(all_files)
    dirs_out = natsorted(dirs_paths_bases)

    LOG.debug(
        "done step=get_filenames path=%s subfolders=%s regex=%s n_files=%d n_dirs=%d time_s=%.3f",
        input_path,
        subfolders,
        regex,
        len(files_out),
        len(dirs_out),
        t.s(),
    )

    return files_out, dirs_out

def match_filenames(list1, list2, trim1=None, trim2=None):
    """
    Match filenames between two lists, optionally trimming a string from the filenames.

    Parameters:
    - list1 (list[str]): First list of filenames.
    - list2 (list[str]): Second list of filenames.
    - trim1 (str, optional): String to trim from filenames in list1 before matching. Default is None.
    - trim2 (str, optional): String to trim from filenames in list2 before matching. Default is None.

    Returns:
    - list[str] or tuple[list[str], list[str]]: A single matched list if matched1 and matched2 are the same,
      otherwise a tuple of two matched lists.
    """
    
    t = Timer()

    if trim1:
        list1 = [f.replace(trim1, '') for f in list1]
    if trim2:
        list2 = [f.replace(trim2, '') for f in list2]
        
    # Perform matching
    matched = set(list1) & set(list2)
    
    # Add trim strings back
    matched1 = [f + (trim1 or '') for f in matched]
    matched2 = [f + (trim2 or '') for f in matched]

    # NOTE: matched is a set -> order is undefined; always sort before comparing.
    out1 = natsorted(matched1)
    out2 = natsorted(matched2)

    LOG.debug(
        "done step=match_filenames n1=%d n2=%d trim1=%s trim2=%s n_matched=%d time_s=%.3f",
        len(list1),
        len(list2),
        trim1,
        trim2,
        len(matched),
        t.s(),
    )

    if out1 == out2:
        return out1
    return out1, out2

def get_project_root():
    try:
        # Try to get git top-level directory
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        # Fallback to current working directory
        root = os.getcwd()
    return root