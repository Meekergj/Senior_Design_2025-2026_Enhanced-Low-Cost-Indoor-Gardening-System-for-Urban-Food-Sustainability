# Copyright info and other stuff
#
#

import tensorflow as tf
import os
from pathlib import Path
import numpy as np
from keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.utils import img_to_array

CURRENT_DIR = Path.cwd()
INPUT_DIR = CURRENT_DIR / "data" / "Test Indices"
EXPORT_DIR = CURRENT_DIR / "data" / "Test Numpy"
image_height = 224
image_width = 224

# Import images of 4 indices that have the same number id from INPUT_DIR
# Example Format: 00001 - dd/mm/yy - NDVI | 00001 - dd/mm/yy - PRI | ...
input_names = []
input_id_index_file = []
for file in os.scandir(INPUT_DIR):
    input_names.append(file.name)

for name in input_names:
    temp = name.replace(" ", "")
    split = temp.split("-")
    if len(split) == 3:
        split[2] = str(split[2]).split(os.extsep)[0] # removes file extension
        input_id_index_file.append([split[0], split[2], name])

four_index_names = []
unique_ids = set([i[0] for i in input_id_index_file])
for uid in unique_ids:
    ndvi = ""
    ari = ""
    pri = ""
    sipi = ""
    for i in input_id_index_file:
        if i[0] == uid and i[1].lower().__eq__("ndvi"):
            ndvi = i[2]
            #input_id_index_file.pop(input_id_index_file.index(i))
        elif i[0] == uid and i[1].lower().__eq__("ari"):
            ari = i[2]
            #input_id_index_file.pop(input_id_index_file.index(i))
        elif i[0] == uid and i[1].lower().__eq__("pri"):
            pri = i[2]
            #input_id_index_file.pop(input_id_index_file.index(i))
        elif i[0] == uid and i[1].lower().__eq__("sipi"): 
            sipi = i[2]
            #input_id_index_file.pop(input_id_index_file.index(i))
        
    if ndvi != "" and ari != "" and pri != "" and sipi != "":
        # This order matters (uid is for keeping track of id for later)!
        four_index_names.append([ndvi, ari, pri, sipi, uid]) 

# Convert images into numpy matrices (greyscaled to make each image 1 layer)
four_index_img = []
for fin in four_index_names:
    try:
        four_index_img.append([[load_img(INPUT_DIR/i, 'grayscale', target_size=(image_height, image_width)) for i in fin[0:4]], fin[4]])
    except:
        print("four_index_img creation did a bad")

# Combine numpy matrices into single 4 layer deep one (same order every time)
four_index_matrices = []
for fii in four_index_img:
    four_arrays = [img_to_array(f) for f in fii[0]]

    # reshape arrays from 3d (x, y, 1) to 2d (x, y)
    shape_prev = np.shape(four_arrays[0])
    four_arrays = [np.reshape(a, (shape_prev[0:2])) for a in four_arrays]

    #four_index_matrices.append([four_arrays, fii[1]])
    four_index_matrices.append([np.stack([four_arrays[0], four_arrays[1], four_arrays[2], four_arrays[3]], axis=2), fii[1]])

#four_index_matrices = [np.stack([b[0][0], b[0][1], b[0][2], b[0][3]], axis=2) for b in four_index_matrices]

# Give 3d numpy matrix name relating to initial number id and save to EXPORT_DIR
for fim in four_index_matrices:
    np.save(file=EXPORT_DIR / str(fim[1]), arr=fim[0])
    