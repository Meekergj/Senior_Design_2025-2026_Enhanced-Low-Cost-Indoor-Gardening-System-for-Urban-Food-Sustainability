# Copyright info and other stuff
#
#

import tensorflow as tf
from pathlib import Path
import numpy as np
from keras import datasets, layers, models
import traceback
import os

CURRENT_DIR = Path.cwd()
EXPORT_DIR = CURRENT_DIR / "tests" / "models"
DATA_DIR = CURRENT_DIR / "data"

commands_str = "1. 'Save' Save Current Model\n" \
               "2. 'Load' Load a Previous Model\n" \
               "3. 'Test' Test an Image\n" \
               "4. 'Train' Train Model using Standard Dataset\n" \
               "5. 'Exit' or 'x' \n"    

def save_model(model):
    model_name = str(input("\nInput the name you want to give the model without the file extention: "))
    file_name = model_name + ".keras"
    export_path = EXPORT_DIR / file_name

    try:
        model.save(export_path)
    except:
        print("\nThat name is FUCKED! (" + str(export_path) + ")\n")
        traceback.print_exc()

def load_model():
    model_name = str(input("\nInput the name of the model without the file extention: "))
    file_name = model_name + ".keras"
    export_path = EXPORT_DIR / file_name

    try:
        return tf.keras.models.load_model(export_path)
    except:
        print("\nThat file does not exist! (" + str(export_path) + ")\n")
        traceback.print_exc()

def test_model(model, image_shape, labels):
    folder_name = str(input("\nInput the name of the folder in the /data/ directory the npy file is inside of: "))
    folder_path = DATA_DIR / folder_name
    npy_name = str(input("\nInput the name of the npy file you want to test on the loaded model: "))
    npy_path = folder_path / npy_name

    try:
        # Load image(s) in the form of a numpy matrix then convert to numpy array for predict method
        loaded_npy = np.load(npy_path)
        image_height = image_shape[0]
        image_width = image_shape[1]
        npy_nparr = tf.keras.preprocessing.image.smart_resize(loaded_npy, (image_height, image_width))
        npy_nparr = np.array([npy_nparr])  # Convert single image to a batch.
        predictions = model.predict(npy_nparr)

        print("Labels: " + str(labels) + "\n")
        print("Predictions: " + str(predictions) + "\n")
    except:
        print("\nThat folder or image file does not exist! (" + str(npy_path) + ")\n")
        print(traceback.format_exc())