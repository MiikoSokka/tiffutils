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
