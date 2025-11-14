# Copyright info and other stuff
#
#

import tensorflow as tf
from pathlib import Path
from keras import datasets, layers, models

CURRENT_DIR = Path.cwd()
EXPORT_DIR = CURRENT_DIR / "tests" / "models"
commands_str = "1. 'Save' Save Current Model\n" \
               "2. 'Load' Load a Previous Model\n" \
               "3. 'Test' Test an Image\n" \
               "4. 'Exit'\n"    

def save_model(model):
    model_name = str(input("\nInput the name of your model without the file extention: "))
    file_name = model_name + ".keras"
    export_path = EXPORT_DIR / file_name
    try:
        model.save(export_path)
    except:
        print("\nThat name is FUCKED! (" + str(export_path) + ")\n")

def load_model():
    model_name = str(input("\nInput the name of the model without the file extention: "))
    file_name = model_name + ".keras"
    export_path = EXPORT_DIR / file_name
    try:
        return tf.keras.models.load_model(export_path)
    except:
        print("\nThat file does not exist! (" + str(export_path) + ")\n")

def test_model(model):
    return ""

# after model is done training and compiling give user option in the console
# to save it, load one, test a model, or exit
def menu(model, history):
    keep_going = True
    while (keep_going == True):
        print(commands_str)
        user_input = input("Enter Command: ")
        match user_input.lower():
            case "save":
                save_model(model)
            case "load":
                model = load_model()
            case "test":
                test_model(model)
            case "exit":
                keep_going = False
                break
            case _:
                print("\nCommand Unrecognized or Misspelled\n")