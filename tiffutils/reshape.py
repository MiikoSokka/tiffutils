
import os
import numpy as np
import tifffile as tiff
from natsort import natsorted


# From process_operaphenixTimepoints.py
#!/usr/bin/env python3

import argparse
import os
import re
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation

print('\n')

def stack_timepoints(input_pattern, channelnumber, read_path):
    # Stack 4D arrays (Z, C, X, Y) into 5D (T, Z, C, X, Y)
    print(input_pattern)
    filenamelist = [file for file in os.listdir(read_path) if os.path.isfile(os.path.join(read_path, file)) and re.match(input_pattern, file)]
    
    filelist_sorted = sorted(filenamelist)
    result_arr = []
    for file in filelist_sorted:
        img = tiff.imread(os.path.join(read_path, file), is_ome=False)
        if len(img.shape) == 3:
            # If the shape of the array is 3D, divide first dimension (Z+C) to create 4D array
            c = channelnumber
            cz = img.shape[0]
            x = img.shape[1]
            y = img.shape[2]
            z = int(cz / c)
            shape = img.shape
            print('Reshaping', file, '\t', shape)
            img = img.reshape(z, c, x, y)
            shape = img.shape
            print('shape after reshaping: ', shape, '\n')
        if img.shape[1] < channelnumber:
            numberofemptychannelsneeded = channelnumber - img.shape[1]
            print(f'File {file} has fewer channels than {channelnumber}. Adding {numberofemptychannelsneeded} empty array(s)...')
            new_array = np.full((19, numberofemptychannelsneeded, 2160, 2160), 255)
            # Concatenate the new empty arrays along the axis 1
            img = np.concatenate((img, new_array), axis=1)
        result_arr.append(img)
    array_5d = np.stack(result_arr, axis=0)
    return array_5d


def normalize_slices(array_5d):

    shape = array_5d.shape
    normalized_array = np.zeros_like(array_5d)
    
    for t in range(shape[0]):
        print('Normalizing timepoint', t)
        for c in range(shape[2]):
            arr = array_5d[t, :, c, :, :]
            array_norm = (arr - arr.min()+1) / (arr.max() - arr.min()+1) * 255
            normalized_array[t, :, c, :, :] = array_norm.astype(np.uint8)
            
    return normalized_array


def align_images(array_5d):
    # Align 5D arrays using 2nd channel as the alignment channel
    shape = array_5d.shape
    aligned_array = np.zeros_like(array_5d)
    
    reference = np.amax(array_5d[0, :, 1, :, :], axis=0)

    for t in range(shape[0]):
        print('Calculating shift for timepoint', t)
        bead_MIP = np.amax(array_5d[t, :, 1, :, :], axis=0)
        
        shift, error, diffphase = phase_cross_correlation(reference, bead_MIP, normalization=None)
        print('Shift:', shift, '\n')
        shift_y, shift_x = map(int, shift)
        
        for z in range(shape[1]):
            for c in range(shape[2]):
                array_5d_rolled = np.roll(array_5d[t, z, c, :, :], (shift_x, shift_y), axis=(1, 0))
                aligned_array[t, z, c, :, :] = array_5d_rolled

    return aligned_array
        

def reshape_timepoints_to_channels(stack):
    t, z, c, y, x = stack.shape
    tc = t * c
    stack = stack.transpose(1, 0, 2, 3, 4).reshape(z, tc, y, x)
    return stack


def save_tiff(stack, filename, output_folder, timepoints_to_channels):
    stack = stack.astype(np.uint8)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set the save path and filename
    filename = os.path.splitext(os.path.basename(filename))[0]
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    # stack = np.squeeze(stack_reshaped)
    
    if timepoints_to_channels:
        stack = reshape_timepoints_to_channels(stack)
    
    save_path = os.path.join(output_folder, f"{filename}.tiff")
    print("Shape of the stack:", stack.shape)
    tiff.imwrite(save_path, stack, imagej=True)
    print("File", f"{filename}.tiff", "saved.")


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(
        prog="stack_timepoints.py",
        description="Stacks 4D tiff images (reshapes 3D to 4D if not reshaped before) into 5D adding timepoints, then aligns the stacks using phase cross correlation on MIP of beads channel 2. Assumes the list of files is in the order of timepoints when sorted.")
    
    # Add arguments
    # parser.add_argument("--filelist", "-f", nargs="+", type=str, help="A list of filenames to be stacked (list items separated by spaces)")
    parser.add_argument("--input_pattern", "-i", type=str, help="A regex pattern to find the input files in folder defined by read-path. Use quotes around the regex. Default = .*\\.tiff?$ to find all .tif or .tiff files", default=".*\\.tiff?$")
    parser.add_argument("--filename", "-n", help="Filename for the aligned stack", default="")
    parser.add_argument("--read_path", "-p", help="Path to the list of files. Default = current folder.", default=".")
    parser.add_argument("--channel_number", "-c", type=int, help="Number of channels in the image.")
    parser.add_argument("--output_folder", "-o", help="Path to the output file. Creates the folder if doesn't exist. Default = current folder.", default=".")
    parser.add_argument("--timepoints_to_channels", "-t", action='store_true', help="Reshape timepoints to channels if set.")
    parser.add_argument("--skip_registration", "-s", action='store_true', help="If set, skips aligning the stack.")

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Stack the images and align
    print(f'Regex pattern is {args.input_pattern}\nFilename for the new stack is {args.filename}\nRead path is {args.read_path}\nChannel number is {args.channel_number}\nOutput folder is {args.output_folder}\n')
    array_5d = stack_timepoints(args.input_pattern, args.channel_number, args.read_path)
    array_5d_normalized = normalize_slices(array_5d)

    if args.skip_registration:
        save_tiff(array_5d_normalized, args.filename, args.output_folder, args.timepoints_to_channels)
    else:
        array_5d_normalized_aligned = align_images(array_5d_normalized)
        save_tiff(array_5d_normalized_aligned, args.filename, args.output_folder, args.timepoints_to_channels)

# From stack_timepoints_from_subfolders.py
#!/usr/bin/env python
# coding: utf-8
# Author: Miiko Sokka

'''
Stack timepoints from subfolders

- This is a modification of stack_timepoints
- Will make a reduced size hyperstack from aligned stacks that are in subfolders
- Will normalize the images using histogram stretching along 0.1–99.999 percentiles
- Will reduce the filesize to uint8
- The purpose is to QC the quality of alignment

'''

# In[1]:


import argparse
import os
import numpy as np
import tifffile as tiff
from natsort import natsorted


# In[5]:


def stack_timepoints(read_path, output_path, channelnumber = 4):
    # Stack 4D arrays (Z, C, X, Y) into 5D (T, Z, C, Y, X)
    
    # Ensure main directory exists
    if not os.path.exists(read_path):
        raise FileNotFoundError(f"Directory not found: {read_path}")
    
    # Get list of subdirectories (assuming they represent timepoints)
    dirs_paths = natsorted([os.path.join(read_path, d) for d in os.listdir(read_path) 
                  if os.path.isdir(os.path.join(read_path, d))])
    
    if not dirs_paths:
        raise ValueError("No subdirectories found in main directory.")
    
    # Ensure all directories have identical filenames
    all_files = os.listdir(dirs_paths[0])  # Assume the first folder has all expected files
    print(f'All files:\n{all_files}')
    
    if not all_files:
        raise ValueError(f"No image files found in: {dirs_paths[0]}")

    for i, fov_name in enumerate(all_files):
        # print(i, fov_name)

        result_array_list = []

        for directory_path in dirs_paths:
            print(f'Processing file {fov_name} in timepoint directory {os.path.basename(os.path.normpath(directory_path))}')

            try:
                # Attempt to read the TIFF file
                array_4D = tiff.imread(os.path.join(directory_path, fov_name), is_ome=False)
            except FileNotFoundError:
                print(f"File {fov_name} not found. Skipping...")
                array_4D = None
            except Exception as e:
                # Handle any other exceptions
                print(f"An error occurred while reading {fov_name}: {e}")
                exit()

            if array_4D is not None:
                print(f'\tShape of array {array_4D.shape}')
    
                if len(array_4D.shape) == 3:
                    array_4D = np.expand_dims(array_4D, axis=1)
                    print(f'\tAssuming array has only one C; expanding dimensions to {array_4D.shape} ')
                    
                if array_4D.shape[1] < channelnumber:
                    numberofemptychannelsneeded = channelnumber - array_4D.shape[1]
                    print(f'File {fov_name} has fewer channels than {channelnumber}. Adding {numberofemptychannelsneeded} empty array(s)...')
                    new_array = np.full((array_4D.shape[0], numberofemptychannelsneeded, array_4D.shape[2], array_4D.shape[3]), 65536)
                    # Concatenate the new empty arrays along the axis 1
                    array_4D = np.concatenate((array_4D, new_array), axis=1)
                
                result_array_list.append(normalize_array(array_4D))

            else:
                continue
                    

            
        array_5D = np.stack(result_array_list, axis=0)
        print(f'Shape of the stacked array is {array_5D.shape} before reshaping...')
        
        # array_4D_timepointStack = reshape_timepoints_to_channels(array_5D)
        # print(f'Shape of the stacked array after reshaping is {array_4D_timepointStack.shape}')

        # Use this line if you want to remove the fiducial beads layers
        save_tiff(np.delete(array_5D, 1, axis=2), fov_name, output_path)

        # Use this line to keep everything 
        # save_tiff(array_5D, fov_name, output_path)


# In[6]:


def reshape_timepoints_to_channels(array_5D):
    t, z, c, y, x = array_5D.shape
    tc = t * c
    array_4D = array_5D.transpose(1, 0, 2, 3, 4).reshape(z, tc, y, x)  # Shape (z, tc, y, x)
    return array_4D


# In[8]:


import numpy as np

def normalize_array(array_4D, intensity_scaling_param=[0.1, 99.999]):
    """
    Normalize a 4D array by scaling intensity values for each color channel separately.

    Parameters:
    - array_4D: 4D NumPy array with shape (Z, C, Y, X)
    - intensity_scaling_param: List containing lower and upper percentile values for intensity scaling

    Returns:
    - stretched_array: Normalized 4D array with intensity values scaled to 0-255 range
    """
    p_lower, p_upper = intensity_scaling_param
    
    # Compute actual percentile values for each color channel
    percentiles = []
    for c in range(array_4D.shape[1]):  # Iterate over color channels
        channel_data = array_4D[:, c, :, :]  # Extract data for current channel
        p1 = np.percentile(channel_data, p_lower)
        p99 = np.percentile(channel_data, p_upper)
        percentiles.append((p1, p99))
    
    print(f'\tMinimum is {np.min(array_4D)}, maximum is {np.max(array_4D)}')
    for c, (p1, p99) in enumerate(percentiles):
        print(f'\tChannel {c}: Lower bound is {p1}, upper bound is {p99}')

    stretched_array = []  # Initialize list to store stretched slices

    for z in range(array_4D.shape[0]):  # Iterate over slices
        stretched_slice = []
        for c in range(array_4D.shape[1]):  # Iterate over color channels
            p1, p99 = percentiles[c]  # Get percentiles for current channel
            stretched_channel = np.clip((array_4D[z, c, :, :] - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            stretched_slice.append(stretched_channel)
        stretched_slice = np.stack(stretched_slice, axis=0)  # Stack channels back
        stretched_array.append(stretched_slice)

    stretched_array = np.stack(stretched_array)  # Convert list to numpy array
    return stretched_array


# In[9]:


def save_tiff(stack, filename, output_folder):
    stack = stack.astype(np.uint8)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    # stack = np.squeeze(stack_reshaped)
    
    save_path = os.path.join(output_folder, f"{filename}")
    print("Shape of the stack:", stack.shape)
    tiff.imwrite(save_path, stack, imagej=True)
    print("File", f"{filename}", "saved.")


# In[10]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description="Makes 5D hyperstacks (T,Z,C,Y,X) from 4D stacks (Z,C,Y,X) that are in subfolders.")
    
    # Add arguments
    parser.add_argument("--input_path", "-i",  help="Path to the root experiment folder, which contains folders for each round of imaging. Default = current folder.", default = ".")
    parser.add_argument("--output_path", "-o", help="Path to the output folder. Creates the folder if doesn't exist. Default = hyperstacks", default="./hyperstacks")
    parser.add_argument("--channel_number", "-c", type=int, help="Number of channels in the image. Default = 4.", default=4)

    args = parser.parse_args()


stack_timepoints(args.input_path, args.output_path, args.channel_number)








# From stack_timepoints_from_hyperstacks.py
#!/usr/bin/env python
# coding: utf-8
# Author: Miiko Sokka

'''
Stack timepoints from subfolders

- This is a modification of stack_timepoints
- Will make a reduced size hyperstack from aligned stacks that are in a single folder, using a user-definable input pattern
- Will normalize the images using histogram stretching along 0.1–99.999 percentiles
- Will reduce the filesize to uint8
- The purpose is to QC the quality of alignment

'''

# In[1]:


import argparse
import os
import re
import numpy as np
import tifffile as tiff
from natsort import natsorted


# In[5]:


def stack_timepoints(input_path, output_path, channel_number, outputfilename, input_pattern='.*\\.tiff?$'):
    # Stack 4D arrays (Z, C, X, Y) into 5D (T, Z, C, Y, X)
    print(f'Using input pattern: {input_pattern}')
    filenamelist = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file)) and re.match(input_pattern, file) and not file.startswith('._')]

    print(f'Stacking files:\n {filenamelist}')
    filelist_sorted = natsorted(filenamelist)
    result_arr = []
    for file in filelist_sorted:
        img = tiff.imread(os.path.join(input_path, file), is_ome=False)
        if len(img.shape) == 3:
            # If the shape of the array is 3D, divide first dimension (Z+C) to create 4D array
            c = channel_number
            cz = img.shape[0]
            x = img.shape[1]
            y = img.shape[2]
            z = int(cz / c)
            shape = img.shape
            print('Reshaping', file, '\t', shape)
            img = img.reshape(z, c, x, y)
            shape = img.shape
            print('shape after reshaping: ', shape, '\n')
        if img.shape[1] < channel_number:
            numberofemptychannelsneeded = channel_number - img.shape[1]
            print(f'File {file} has fewer channels than {channel_number}. Adding {numberofemptychannelsneeded} empty array(s)...')
            new_array = np.full((img.shape[0], numberofemptychannelsneeded, img.shape[2], img.shape[3]), 65536)
            # Concatenate the new empty arrays along the axis 1
            img = np.concatenate((img, new_array), axis=1)
        result_arr.append(img)
    array_5d = np.stack(result_arr, axis=0)

    save_tiff(array_5d, outputfilename, output_path)

    return array_5d


def reshape_timepoints_to_channels(array_5D):
    t, z, c, y, x = array_5D.shape
    tc = t * c
    array_4D = array_5D.transpose(1, 0, 2, 3, 4).reshape(z, tc, y, x)  # Shape (z, tc, y, x)
    return array_4D


# In[8]:


import numpy as np

def normalize_array(array_4D, intensity_scaling_param=[0.1, 99.999]):
    """
    Normalize a 4D array by scaling intensity values for each color channel separately.

    Parameters:
    - array_4D: 4D NumPy array with shape (Z, C, Y, X)
    - intensity_scaling_param: List containing lower and upper percentile values for intensity scaling

    Returns:
    - stretched_array: Normalized 4D array with intensity values scaled to 0-255 range
    """
    p_lower, p_upper = intensity_scaling_param
    
    # Compute actual percentile values for each color channel
    percentiles = []
    for c in range(array_4D.shape[1]):  # Iterate over color channels
        channel_data = array_4D[:, c, :, :]  # Extract data for current channel
        p1 = np.percentile(channel_data, p_lower)
        p99 = np.percentile(channel_data, p_upper)
        percentiles.append((p1, p99))
    
    print(f'\tMinimum is {np.min(array_4D)}, maximum is {np.max(array_4D)}')
    for c, (p1, p99) in enumerate(percentiles):
        print(f'\tChannel {c}: Lower bound is {p1}, upper bound is {p99}')

    stretched_array = []  # Initialize list to store stretched slices

    for z in range(array_4D.shape[0]):  # Iterate over slices
        stretched_slice = []
        for c in range(array_4D.shape[1]):  # Iterate over color channels
            p1, p99 = percentiles[c]  # Get percentiles for current channel
            stretched_channel = np.clip((array_4D[z, c, :, :] - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            stretched_slice.append(stretched_channel)
        stretched_slice = np.stack(stretched_slice, axis=0)  # Stack channels back
        stretched_array.append(stretched_slice)

    stretched_array = np.stack(stretched_array)  # Convert list to numpy array
    return stretched_array


# In[9]:


def save_tiff(stack, filename, output_folder):
    stack = stack.astype(np.uint8)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    # stack = np.squeeze(stack_reshaped)
    
    save_path = os.path.join(output_folder, f"{filename}")
    print("Shape of the stack:", stack.shape)
    tiff.imwrite(save_path, stack, imagej=True)
    print("File", f"{filename}", "saved.")


# In[10]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description="Makes 5D hyperstacks (T,Z,C,Y,X) from 4D stacks (Z,C,Y,X) that are in subfolders.")
    
    # Add arguments

    parser.add_argument("--input_path", "-i",  help="Path to the root experiment folder, which contains folders for each round of imaging. Default = current folder.", default = ".")
    parser.add_argument("--output_path", "-o", help="Path to the output folder. Creates the folder if doesn't exist. Default = current folder", default=".")
    parser.add_argument("--channel_number", "-c", type=int, help="Number of channels in the image. Default = 4.", default=4)
    parser.add_argument("--outputfilename", "-f", help="Filename for the aligned stack", default="hyperstack.tiff")
    parser.add_argument("--input_pattern", "-p", type=str, help="A regex pattern to find the input files in folder defined by read-path. Use quotes around the regex. Default = .*\\.tiff?$ to find all .tif or .tiff files", default=".*\\.tiff?$")
    args = parser.parse_args()


stack_timepoints(args.input_path, args.output_path, args.channel_number, args.outputfilename, args.input_pattern)








# From stack_subsetOfRawOPimages.py
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



# From stack_CYX_to_TCYX.py
#!/usr/bin/env python
# coding: utf-8
# Author: Miiko Sokka

'''
Stack timepoints from subfolders

- This is a modification of stack_timepoints
- Will make a reduced size hyperstack from aligned stacks that are in a single folder, using a user-definable input pattern
- Will normalize the images using histogram stretching along 0.1–99.999 percentiles
- Will reduce the filesize to uint8
- The purpose is to QC the quality of alignment

'''

# In[1]:


import argparse
import os
import re
import numpy as np
import tifffile as tiff
from natsort import natsorted


def mip_tif(input_path, output_path, input_pattern='.*\\.tiff?$'):
    # Stack 4D arrays (Z, C, X, Y) into 5D (T, Z, C, Y, X)
    print(f'Using input pattern: {input_pattern}')
    filenamelist = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file)) and re.match(input_pattern, file) and not file.startswith('._')]

    print(f'MIPping files:\n {filenamelist}')
    filelist_sorted = natsorted(filenamelist)
    print(f'Filelist: {filelist_sorted}')

    stack_list= []

    for file in filelist_sorted:

        img = tiff.imread(os.path.join(input_path, file), is_ome=False)
        shape = img.shape
        print(f'Processing file {file} with a shape {shape}')

        if img.ndim != 3:
            raise ValueError(f"Image {fname} does not have 3 dimensions (C, Y, X).")
        
        stack_list.append(img)

    # Convert to TCYX numpy array
    tcyx_array = np.stack(stack_list, axis=0)  # shape: (T, C, Y, X)

    save_tiff(tcyx_array, 'hyperstack.tiff', output_path)


def save_tiff(stack, filename, output_folder):
    stack = stack.astype(np.uint8)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    # stack = np.squeeze(stack_reshaped)
    
    save_path = os.path.join(output_folder, f"{filename}")
    print("Shape of the stack:", stack.shape)
    tiff.imwrite(save_path, stack, imagej=True)
    print("File", f"{filename}", "saved.")


# In[10]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description="Makes 5D hyperstacks (T,Z,C,Y,X) from 4D stacks (Z,C,Y,X) that are in subfolders.")
    
    # Add arguments

    parser.add_argument("--input_path", "-i",  help="Path to the root experiment folder, which contains folders for each round of imaging. Default = current folder.", default = ".")
    parser.add_argument("--output_path", "-o", help="Path to the output folder. Creates the folder if doesn't exist. Default = .", default=".")
    parser.add_argument("--input_pattern", "-p", type=str, help="A regex pattern to find the input files in folder defined by read-path. Use quotes around the regex. Default = .*\\.tiff?$ to find all .tif or .tiff files", default=".*\\.tiff?$")
    args = parser.parse_args()


mip_tif(args.input_path, args.output_path, args.input_pattern)



# From reshape_timepoints-to-channels.py
#!/usr/bin/env python3

import argparse
import os
import re
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation

print('\n')


def read_tif(file, read_path):
    img_array = tiff.imread(os.path.join(read_path, file), is_ome=False)
    stack = reshape_timepoints_to_channels(img_array)
    return stack
    

def reshape_timepoints_to_channels(stack):
    t, z, c, y, x = stack.shape
    tc = t * c
    stack = stack.transpose(1, 0, 2, 3, 4).reshape(z, tc, y, x)
    return stack


def save_tiff(stack, filename, output_folder):
    stack = stack.astype(np.uint8)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set the save path and filename
    filename = os.path.splitext(os.path.basename(filename))[0]
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    # stack = np.squeeze(stack_reshaped)
    
    save_path = os.path.join(output_folder, f"{filename}.tiff")
    print("Shape of the stack:", stack.shape)
    tiff.imwrite(save_path, stack, imagej=True)
    print("File", f"{filename}.tiff", "saved.")


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(
        prog="reshape_timepoints-to-channels.py",
        description="Reshapes tiff image.")
    
    # Add arguments
    parser.add_argument("--input_file", "-i", type=str, help="A regex pattern to find the input files in folder defined by read-path. Use quotes around the regex. Default = .*\\.tiff?$ to find all .tif or .tiff files", default=".*\\.tiff?$")
    parser.add_argument("--output_filename", "-n", help="Base filename for the reshaped stack", default="reshaped_stack")
    parser.add_argument("--read_path", "-p", help="Path to the input file. Default = current folder.", default=".")
    parser.add_argument("--output_folder", "-o", help="Path to the output file. Creates the folder if doesn't exist. Default = current folder.", default=".")

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Stack the images and align
    img = read_tif(args.input_file, args.read_path)
    save_tiff(img, args.output_filename, args.output_folder)
