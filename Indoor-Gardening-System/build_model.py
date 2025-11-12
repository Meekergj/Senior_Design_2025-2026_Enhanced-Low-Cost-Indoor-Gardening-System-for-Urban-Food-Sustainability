# Copyright info and other stuff
#
#

import tensorflow as tf

from keras import datasets, layers, models
import matplotlib.pyplot as plt

# Get images and labels as training and validation data
def load(data_path, image_height, image_width):
    # From the folder data_path get images from subfolders (labels = subfolder name)
    train_set = tf.keras.utils.image_dataset_from_directory(
                data_path,
                validation_split=0.3,
                subset="training",
                seed=448,
                image_size=(image_height, image_width))
    
    validation_set = tf.keras.utils.image_dataset_from_directory(
                data_path,
                validation_split=0.3,
                subset="validation",
                seed=448,
                image_size=(image_height, image_width))
    
    # Tensorflow magic to cache and prefetch this data, increase performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

    # convert pixels from [0 255] to [0 1], easier on neural net
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_set = train_set.map(lambda x, y: (normalization_layer(x), y))

    return normalized_set, validation_set

# Convolutional Neural Net
def build(train_set, validation_set, input_shape, num_classes):
    model = models.Sequential()

    # Slightly randomizes the input to prevent overfitting and increase accuracy
    model.add(layers.Input(input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomZoom(0.2))

    # Layers of the CNN model
    model.add(layers.Conv2D(16, 5, activation='relu', input_shape=input_shape, padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 5, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(72, 3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))

    # Take the aformentioned layers of the model and compile together
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    return model
