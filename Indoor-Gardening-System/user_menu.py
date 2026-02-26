# Copyright info and other stuff
#
#

import tensorflow as tf
from pathlib import Path
import numpy as np
from keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.utils import img_to_array
import traceback

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

def test_model(model, shape, labels):
    folder_name = str(input("\nInput the name of the folder the image is inside of (excluding the path): "))
    folder_path = DATA_DIR / folder_name
    image_name = str(input("\nInput the name of the image file you want to test on the loaded model: "))
    image_path = folder_path / image_name

    try:
        # Load image as PIL then convert to numpy array for predict method
        loaded_image = load_img(image_path, target_size=shape)
        image_nparr = img_to_array(loaded_image)
        image_nparr = np.array([image_nparr])  # Convert single image to a batch.
        predictions = model.predict(image_nparr)

        highest_prediction = np.argmax(predictions[0])
        print(labels[highest_prediction] + "\n")
    except:
        print("\nThat folder or image file does not exist! (" + str(image_path) + ")\n")
        traceback.print_exc()