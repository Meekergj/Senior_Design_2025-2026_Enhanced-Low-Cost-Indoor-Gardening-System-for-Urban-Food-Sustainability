# Copyright info and other stuff
#
#

import tensorflow as tf

from keras import datasets, layers, models
import build_model as bm
import matplotlib.pyplot as plt
from pathlib import Path

#-----------------------------------------------------------#
# Variables / Parameters
#-----------------------------------------------------------#
image_height = 128
image_width = 128
shape = (image_height, image_width, 3)
labels = ["healthy", "diseased"]
labels_size = len(labels)

CURRENT_DIR = Path.cwd()
DATA_DIR = CURRENT_DIR / "data" / "Gauva (P3)"
CHECKPOINT_DIR = CURRENT_DIR / "models" / "checkpoints"
EXPORT_DIR = CURRENT_DIR / "tests" / "models"

#-----------------------------------------------------------#
# Build and Compile Model
#-----------------------------------------------------------#
#data_path = tf.keras.utils.get_file(DATA_DIR, extract=True)
#data_path = Path(data_path).with_suffix('')
train_set, validation_set = bm.load(DATA_DIR, image_height, image_width)
model = bm.build(train_set, validation_set, shape, labels_size)

#-----------------------------------------------------------#
# Train Model
#-----------------------------------------------------------#
epochs = 10
history = model.fit(
  train_set,
  validation_data=validation_set,
  epochs=epochs
)

#-----------------------------------------------------------#
# Visualize (from https://www.tensorflow.org/tutorials/images/classification)
#-----------------------------------------------------------#
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()