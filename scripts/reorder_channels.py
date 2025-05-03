#!/usr/bin/env python3

import argparse
import os
import sys
import tifffile as tiff
import numpy as np

def print_shape(filename, flag):
    img = tiff.imread(filename, is_ome=False)
    shape = img.shape
    if flag:
        print("\nThe image shape (Z, C, Y, X) is: ", shape, "\n")
        sys.exit()
    else:
        global num_slices
        global num_channels
        num_slices = shape[0]
        num_channels = shape[1]

def save_tiff(stack, filename, prefix, suffix, output_folder):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set the save path and filename
    filename = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(output_folder, f"{prefix}{filename}{suffix}.tiff")
    # Remove all axes with dimension 1 from the array
    slices = stack.shape[0]
    stack = np.squeeze(stack)
    # Save the stacked image as a new TIFF
    tiff.imwrite(save_path, stack, imagej=True) # removed dtype='uint8'
    print("File", f"{prefix}{filename}{suffix}.tiff", "saved.")


def reshape_channels(filename, channel_order):
    img = tiff.imread(filename, is_ome=False)
    img_reshaped = img[:, [channel_order], :, :]
    print("Reshaped the channels of ", filename, "to ", channel_order)
    return img_reshaped
        

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(
        prog="reorder_channels.py",
        description="Reorders the channels in a ZCYX tiff hyperstack.")
    
    # Add arguments
    parser.add_argument("filename", type=str, help="Input filename")
    parser.add_argument("--channel_order", "-co", nargs="+", type=int, help="Reorganize the channels using a list of index integers from 0 to N, where N is the number of channels. The index number refers to the channel to be relocated to that position. E.g. 2 0 3 1 will move 3rd channel to first, 1st channel to second, 4th channel to third and 2nd channel to fourth position")
    parser.add_argument("--prefix", "-p", help="Prefix to be added to the filename", default = "")
    parser.add_argument("--suffix", "-s", help="Suffix to be added to the filename. Default =  _reordered.", default = "_reordered")
    parser.add_argument("--shape", action="store_true", help="Print the shape (Z, C, Y, X) of the image and exit.")
    parser.add_argument("--output", "-o", help="Path for the output file. Default = current folder.", default=".")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the argument values
    filename = args.filename
    channel_order = args.channel_order
    prefix = args.prefix
    suffix = args.suffix
    flag = args.shape
    output_folder = args.output
    
    # Call the print shape function with the boolean argument and exit
    print_shape(filename, flag)

    # Call the functions to reshape the channels and save the new file
    stack_reshaped = reshape_channels(filename, channel_order)
    save = save_tiff(stack_reshaped, filename, prefix, suffix, output_folder)
