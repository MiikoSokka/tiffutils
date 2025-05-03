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






