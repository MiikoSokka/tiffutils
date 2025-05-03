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






