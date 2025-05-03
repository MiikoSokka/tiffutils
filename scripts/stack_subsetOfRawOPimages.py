#!/usr/bin/env python
# coding: utf-8

'''
This script takes in a tsv table of Opera Phenix raw tiff files to be stacked for 3D (keeps the imaging rounds in separate folders).

Designed to process chromosome spread images, which don't necessarily have the same filename for different imaging rounds.
The FOVs need to be matched based on their xy coordinates.

Input:
    filelistToProcess = a tsv file (with a header), three columns: 
        FOVid, an assigned ID that matches FOV images from different rounds)
        Folder, the name of the timepoint folder
        Filename, the name of the filename of form r##c##f##
    imageBasePath = path to the basefolder where raw OP images are (assumes images are in subfolder <timepointfolder>/Images)
    outputPath = base folder path for the stacks to be saved. Creates subfolders for each imaging round/timepoint

Processing:
    Simply creates a tiff stack of shape ZCYX from raw OP tiff files (each z slice, channel separate file) using a list matched filenames in different imaging rounds.
    No other processing (such as normalization, registration) is done.
Output:
    Saves ZCYX tiff stacks into a separate folder, using subfolders for each imaging round (timepoint)
'''

# In[ ]:


import tifffile as tiff
import numpy as np
import os
import pandas as pd
from pathlib import Path
import argparse


# In[ ]:


def reshape_array(array, channelnumber=4):
    """
    Reshape the array and adjust metadata accordingly.
    
    Args:
    - channelnumber (int): Number of channels to split.
    
    Returns:
    - ArrayWithMetadata: The reshaped array with modified metadata.
    """
    img = array
    c = channelnumber
    cz = img.shape[0]
    x = img.shape[1]
    y = img.shape[2]
    z = int(cz / c)
    shape = img.shape
    img_reshaped = img.reshape(z, c, x, y)
    print(f"\tReshaped the array from {shape} to {img_reshaped.shape}")
    
    
    return img_reshaped


# In[ ]:


def stack_raw_operaphenix_tifffiles(timepointpath, identifier, filename_base):
    """
    Stacks raw tiff files from Opera Phenix into ZCYX tiff stack.
    
    Input: A base filename (r##c##f###) corresponding to all raw Opera Phenix tiff files to be stacked.
    Operations: Loads them as numpy arrays, stacks them, and applies various operations (reshaping, MIP, normalization).
    Returns: List of ImageDataWithMetadata objects containing stacked, reshaped, MIPped, normalized numpy array and metadata.
    """
    
    # Sets the path where tiff files are found
    image_path = os.path.join(timepointpath, 'Images')
    print(f'Image Path = {image_path}')
    
    # Get all the files matching the base filename (filename_base)
    all_files = [f for f in os.listdir(image_path) if f.startswith(filename_base + 'p') and f.endswith('.tiff')]
    if not all_files:
        print(f"No files found for {filename_base} in {image_path}")
        return None
        
    all_files.sort()  # Sort files alphabetically
    
    # Stack images
    stack = []
    for file in all_files:
        
        file_path = os.path.join(image_path, file)
        array = tiff.imread(file_path)
        
        stack.append(array)
    
    if not stack:
        print(f"No images were stacked for {filename_base}")
        return None  # Handle case where no images were found
    
    # Stack the arrays and associate metadata
    stacked_array = np.stack(stack, axis=0)
    
    # Reshape and MIP operations can now be chained
    stacked_array_reshaped = reshape_array(stacked_array, channelnumber=4)
    print(f'stacked_array_reshaped {stacked_array_reshaped.shape}')
    return stacked_array_reshaped


# In[ ]:


def process_filelist(filelist_path, read_path, output_folder):
        
    # Load the TSV file into a DataFrame
    df = pd.read_csv(filelist_path, sep='\t')
    
    # Sort by Identifier and Folder
    df = df.sort_values(by=['Identifier', 'Folder'])


    
    # Process each Identifier
    for identifier in df['Identifier'].unique():
        identifier_str = str(identifier).zfill(4)
        print(f"Processing FOVid: {identifier_str}")
        
        # Get rows related to the current Identifier
        identifier_df = df[df['Identifier'] == identifier]
        
        # Prepare file paths and metadata for stacking
        for _, row in identifier_df.iterrows():
            timepointfolder = row['Folder']

            ''' This block defines the output paths'''
            # Define the output path for the TIFF file
            timepointpath = timepointfolder.split('-')[-1]
            output_path = os.path.join(output_folder, timepointpath)
    
            if os.path.exists(os.path.join(output_path, f'FOVid_{identifier_str}.tiff')):
                print(f"Output file already exists for Identifier {identifier_str}, skipping processing.")
                continue

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            filename_base = row['Filename']
            if pd.isna(filename_base) or filename_base == 'NA':
                print(f"Skipping row with invalid or missing Filename in Folder: {timepointfolder}")
                continue
            
            timepointpath = os.path.join(read_path, timepointfolder)
            
            # Stack the TIFF files for the current timepointfolder and filename_base
            stacked_array = stack_raw_operaphenix_tifffiles(timepointpath, identifier_str, filename_base)

            # Write the array into tiff file
            tiff.imwrite(os.path.join(output_path, f'FOVid_{identifier_str}.tiff'), stacked_array, dtype='uint16')
            print(f"Saved concatenated stack for Identifier {identifier_str} to {output_path}\n")


# In[ ]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description="This script takes in a tsv table of Opera Phenix raw tiff files to be stacked for 3D (keeps the imaging rounds in separate folders).")
    
    # Add arguments
    parser.add_argument("--matched_files_table", "-t",  help="Name of the tsv file that contains matched files information, FOVid|Folder|Filename")
    parser.add_argument("--image_base_folder", "-i", help="Path to the base folder where raw images are", default=".")
    parser.add_argument("--output_folder", "-o", help="Path to the output folder. Creates the folder if doesn't exist. Default = stacks.", default="stacks")

    args = parser.parse_args()

    process_filelist(args.matched_files_table, args.image_base_folder, args.output_folder)

