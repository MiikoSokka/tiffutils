# coding: utf-8
# Author: Miiko Sokka

import os
import re
from natsort import natsorted

def get_filenames(input_path, regex=r'.*\.tiff?$', subfolders=False):
    """
    Retrieve a naturally sorted list of filenames matching a given regex pattern from a directory.

    Parameters:
    - input_path (str): Path to the main directory.
    - regex (str): Regular expression to match filenames. Default is '.*\\.tiff?$'.
    - subfolders (bool): If False, search only in the main folder. If True, require subdirectories and validate single matching file across them.

    Returns:
    - list[str]: A naturally sorted list of filenames found (only file names, not full paths).

    Raises:
    - FileNotFoundError: If the input_path does not exist.
    - ValueError: If subfolders=True and no subdirectories are found.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Directory not found: {input_path}")

    if not subfolders:
        filenamelist = [
            file for file in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, file)) and
               re.match(regex, file) and
               not file.startswith('._')
        ]
        return natsorted(filenamelist)
    
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
           re.match(regex, f) and
           not f.startswith('._')
    ]
    print(f'All files:\n{all_files}')
    return natsorted(all_files)