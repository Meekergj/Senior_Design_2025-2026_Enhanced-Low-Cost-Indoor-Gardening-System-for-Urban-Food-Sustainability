# Copyright info and other stuff
#
#

import tensorflow as tf

from keras import datasets, layers, models
import build_model as bm
import user_menu as um
import matplotlib.pyplot as plt
from pathlib import Path

#-----------------------------------------------------------#
# Variables / Parameters
#-----------------------------------------------------------#
image_height = 224
image_width = 224
shape = (image_height, image_width, 3)
labels = ["healthy", "diseased"]
labels_size = len(labels)
batch_size = None
epochs = 10

CURRENT_DIR = Path.cwd()
DATA_DIR = CURRENT_DIR / "data" / "Gauva (P3)"
CHECKPOINT_DIR = CURRENT_DIR / "models" / "checkpoints"
checkpoint_path = CHECKPOINT_DIR / "checkpoint.model.keras"
EXPORT_DIR = CURRENT_DIR / "tests" / "models"

#-----------------------------------------------------------#
# Build and Compile Model
#-----------------------------------------------------------#
#data_path = tf.keras.utils.get_file(DATA_DIR, extract=True)
#data_path = Path(data_path).with_suffix('')
train_set, validation_set = bm.load(DATA_DIR, image_height, image_width, batch_size)
model = bm.build(train_set, validation_set, shape, labels_size)

#-----------------------------------------------------------#
# Train Model
#-----------------------------------------------------------#
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
  train_set,
  validation_data=validation_set,
  epochs=epochs,
  callbacks=cp_callback,
  #batch_size=batch_size
)

tf.keras.models.load_model(checkpoint_path)

#-----------------------------------------------------------#
# Visualize (from https://www.tensorflow.org/tutorials/images/classification)
#-----------------------------------------------------------#
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

sparse_categorical = history.history['sparse_categorical_crossentropy']
val_sparse_categorical = history.history['val_sparse_categorical_crossentropy']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(2, 2, 3)
plt.plot(epochs_range, sparse_categorical, label='Training Cross Entropy')
plt.plot(epochs_range, val_sparse_categorical, label='Validation Cross Entropy')
plt.legend(loc='lower left')
plt.title('Training and Validation Cross Entropy')
plt.show()

um.menu(model, history)