
import os
import numpy as np
import tifffile as tiff
from natsort import natsorted


# From MIP_from_tiffs_in_folder.py
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


def mip_tif(input_path, output_path, channel_number, input_pattern='.*\\.tiff?$'):
    # Stack 4D arrays (Z, C, X, Y) into 5D (T, Z, C, Y, X)
    print(f'Using input pattern: {input_pattern}')
    filenamelist = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file)) and re.match(input_pattern, file) and not file.startswith('._')]

    print(f'MIPping files:\n {filenamelist}')
    filelist_sorted = natsorted(filenamelist)
    print(f'Filelist: {filelist_sorted}')

    for file in filelist_sorted:

        img_array = tiff.imread(os.path.join(input_path, file), is_ome=False)
        shape = img_array.shape
        print(f'Processing file {file} with a shape {shape}')

        mip_img_array= []
        
        if len(shape) == 3:
            # If the shape of the array is 3D, divide first dimension (Z+C) to create 4D array
            c = channel_number
            cz = img_array.shape[0]
            x = img_array.shape[1]
            y = img_array.shape[2]
            z = int(cz/c)
            print("Reshaping", file, shape)
            img_array = img_array.reshape(z, c, x, y)
            shape = img_array.shape
            print(file, "shape after reshaping: ", shape)
            print(img_array.shape)
            stack_mip(img_array)

        elif len(shape) == 4:
            for channel in range(shape[1]):
                # Perform Maximum Intensity Projection along the Z-axis
                mip_image = np.amax(img_array[:, channel, :, :], axis=0)
                mip_img_array.append(mip_image)
               
        elif len(shape) == 5:
            for timepoint in range(shape[0]):
                for channel in range(shape[2]):
                    mip_image = np.amax(img_array, axis=1)
                    mip_img_array.append(mip_image)
        
        else:
            print("Exiting... Stack dimension something else than 3D, 4D or 5D. Check your tiff stack.")
            sys.exit()
            
        mip_img_array = np.stack(mip_img_array)

        save_tiff(mip_img_array, file, output_path)


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
    parser.add_argument("--output_path", "-o", help="Path to the output folder. Creates the folder if doesn't exist. Default = ./mip", default="./mip")
    parser.add_argument("--channel_number", "-c", type=int, help="Number of channels in the image. Default = 4.", default=4)
    parser.add_argument("--input_pattern", "-p", type=str, help="A regex pattern to find the input files in folder defined by read-path. Use quotes around the regex. Default = .*\\.tiff?$ to find all .tif or .tiff files", default=".*\\.tiff?$")
    args = parser.parse_args()


mip_tif(args.input_path, args.output_path, args.channel_number, args.input_pattern)

