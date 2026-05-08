# Filename: main.py
# Author: Gavin Meeker
# Created: 2025-10-15
# Description: File where code is executed and primary variables and model
#              parameters are set.

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

import build_model as bm
import user_menu as um

#-----------------------------------------------------------#
# Variables / Parameters
#-----------------------------------------------------------#
image_height = 224
image_width = 224
num_indices = 3
shape = (image_height, image_width, num_indices, )
labels = ["Hydration", "Nutrition", "Lighting"]
num_labels = len(labels)
batch_size = 3
epochs = 50

CURRENT_DIR = Path.cwd()
TEST_DATA_DIR = CURRENT_DIR / "data" / "Gauva (P3)"
CHECKPOINT_DIR = CURRENT_DIR / "models" / "checkpoints"
checkpoint_path = CHECKPOINT_DIR / "checkpoint.model.keras"
EXPORT_DIR = CURRENT_DIR / "tests" / "models"
LABEL_DIR = CURRENT_DIR / "data" / "Labels"
NPY_DIR = CURRENT_DIR / "data" / "Numpy"

#-----------------------------------------------------------#
# Build and Compile Model
#-----------------------------------------------------------#
train_set, validation_set = bm.load(NPY_DIR, LABEL_DIR, image_height, image_width, batch_size)
model = bm.build(shape, num_labels)

#-----------------------------------------------------------#
# Train Model
#-----------------------------------------------------------#
def train(model, train_set, validation_set, epochs): 
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      monitor='val_mse', 
      mode='min', #minimize loss or maximize accuracy
      save_best_only=True)

  history = model.fit(
    x=train_set,
    validation_data=validation_set,
    epochs=epochs,
    callbacks=cp_callback,
    batch_size=batch_size,
    verbose=2
  )

  tf.keras.models.load_model(checkpoint_path)
  return history

#-----------------------------------------------------------#
# Visualize (from https://www.tensorflow.org/tutorials/images/classification)
#-----------------------------------------------------------#
def visualize_training(history, epochs):
    # Total model loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Individual output MSE metrics
    hydration_mse = history.history['hydration_mse']
    val_hydration_mse = history.history['val_hydration_mse']

    nutrition_mse = history.history['nutrition_mse']
    val_nutrition_mse = history.history['val_nutrition_mse']

    lighting_mse = history.history['lighting_mse']
    val_lighting_mse = history.history['val_lighting_mse']

    epochs_range = range(epochs)

    plt.figure(figsize=(14, 10))

    # Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Total Training and Validation Loss')

    # Hydration MSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, hydration_mse, label='Training Hydration MSE')
    plt.plot(epochs_range, val_hydration_mse, label='Validation Hydration MSE')
    plt.legend(loc='upper right')
    plt.title('Hydration MSE')

    # Nutrition MSE
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, nutrition_mse, label='Training Nutrition MSE')
    plt.plot(epochs_range, val_nutrition_mse, label='Validation Nutrition MSE')
    plt.legend(loc='upper right')
    plt.title('Nutrition MSE')

    # Lighting MSE
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, lighting_mse, label='Training Lighting MSE')
    plt.plot(epochs_range, val_lighting_mse, label='Validation Lighting MSE')
    plt.legend(loc='upper right')
    plt.title('Lighting MSE')

    plt.show()

# after model is done training and compiling give user option in the console
# to save it, load one, test a model, or exit
def menu(model):
    keep_going = True
    while (keep_going == True):
        print(um.commands_str)
        user_input = input("Enter Command: ")
        match user_input.lower():
            case "save":
                um.save_model(model)
            case "load":
                model = um.load_model()
            case "test":
                um.test_model(model, shape, labels)
            case "train":
              history = train(model, train_set, validation_set, epochs)
              visualize_training(history, epochs)
            case "exit":
                keep_going = False
                break
            case "x":
                keep_going = False
                break
            case _:
                print("\nCommand Unrecognized or Misspelled\n")

menu(model)
