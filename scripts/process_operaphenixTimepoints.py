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