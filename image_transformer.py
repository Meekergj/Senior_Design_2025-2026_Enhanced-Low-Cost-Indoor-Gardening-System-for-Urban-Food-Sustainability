# Copyright info and other stuff
#
#

import os
from pathlib import Path
import numpy as np
from keras.preprocessing.image import load_img
from keras.utils import img_to_array

CURRENT_DIR = Path.cwd()
INPUT_DIR = CURRENT_DIR / "data" / "Indices"
EXPORT_DIR = CURRENT_DIR / "data" / "Numpy"
image_height = 224
image_width = 224

# Import images of N indices that have the same number id from INPUT_DIR
# Example Format: 00001 - dd/mm/yy - NDVI | 00001 - dd/mm/yy - PRI | ...
def files_to_names(input_dir):
    input_names = []
    input_id_index_name = []
    for file in os.scandir(input_dir):
        input_names.append(file.name)

    for name in input_names:
        temp = name.replace(" ", "")
        split = temp.split("-")
        if len(split) == 3:
            split[2] = str(split[2]).split(os.extsep)[0] # removes file extension
            file_id = split[0]
            file_index = split[2]
            input_id_index_name.append([file_id, file_index, name])

    index_names = []
    unique_ids = set([i[0] for i in input_id_index_name]) # makes sure each id has all of its indices checked
    for uid in unique_ids:
        rgb = ""
        gndvi = ""
        sipi = ""
        for i in input_id_index_name:
            if i[0] == uid and i[1].lower().__eq__("rgb"):
                rgb = i[2]
            elif i[0] == uid and i[1].lower().__eq__("sipi"):
                sipi = i[2]
            elif i[0] == uid and i[1].lower().__eq__("gndvi"):
                gndvi = i[2]
            
        if rgb != ""  and sipi != "" and gndvi != "":
            # This order matters (uid is for keeping track of id for later)!
            index_names.append([rgb, sipi, gndvi, uid])
    return index_names

# Import images of N indices that have the same number id from INPUT_DIR, but only ndvi
def files_to_names_gndvi(input_dir):
    input_names = []
    input_id_index_name = []
    for file in os.scandir(input_dir):
        input_names.append(file.name)

    for name in input_names:
        temp = name.replace(" ", "")
        split = temp.split("-")
        if len(split) == 3:
            split[2] = str(split[2]).split(os.extsep)[0] # removes file extension
            file_id = split[0]
            file_index = split[2]
            input_id_index_name.append([file_id, file_index, name])

    index_names = []
    unique_ids = set([i[0] for i in input_id_index_name]) # makes sure each id has all of its indices checked
    for uid in unique_ids:
        gndvi = ""
        for i in input_id_index_name:
            if i[0] == uid and i[1].lower().__eq__("gndvi"):
                    ndvi = i[2]
            
        if ndvi != "":
            # This order matters (uid is for keeping track of id for later)!
            index_names.append([ndvi, uid])
    return index_names

# Convert images into numpy matrices (greyscaled to make each image 1 layer)
def images_to_npy(index_names, input_dir, export_dir):
    multi_index_img = []
    for fin in index_names:
        try:
            # fin[-1] is the unique id at end of array
            multi_index_img.append([[load_img(input_dir/i, 'grayscale', target_size=(image_height, image_width)) for i in fin[0:-1]], fin[-1]])
        except:
            print("multi_index_img creation did a bad")

    # Combine numpy matrices into single multi layer deep one (same order every time)
    multi_index_matrices = []
    for mii in multi_index_img:
        multi_arrays = [img_to_array(f) for f in mii[0]]

        # reshape arrays from 3d (x, y, 1) to 2d (x, y)
        shape_prev = np.shape(multi_arrays[0])
        multi_arrays = [np.reshape(a, (shape_prev[0:2])) for a in multi_arrays]

        #multi_index_matrices.append([multi_arrays, mii[1]])
        multi_index_matrices.append([np.stack(multi_arrays[0:], axis=2), mii[1]])

    # Give 3d numpy matrix name relating to initial number id and save to EXPORT_DIR
    for mim in multi_index_matrices:
        np.save(file=export_dir / str(mim[1]), arr=mim[0])

index_names = files_to_names(input_dir=INPUT_DIR)
images_to_npy(index_names, input_dir=INPUT_DIR, export_dir=EXPORT_DIR)
