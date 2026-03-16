# Copyright info and other stuff
#
#

import tensorflow as tf

from keras import datasets, layers, models, metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Helper method, places npy files into single (resized) matrix
def npy_to_matrix(npy_path, image_height, image_width):
    out = [] # rough format (matrixN[file info, file ID], ...)
    for file in os.scandir(npy_path):
        loaded_np = np.load(npy_path / file.name)
        out.append([tf.keras.preprocessing.image.smart_resize(loaded_np, (image_height, image_width)), str(file.name.split(os.extsep)[0])])
    out = {id: matrix for matrix, id in out} # format {key: ID | value: image matrix}
    return out

# Helper method, gets IDs/labels from csv file and places into dictionary (exclues ids not in examples)
def csv_labels(labels_path, examples_keys):
    csv_file_name = 'Hydroponic Bot Test Sheet.csv'

    out = [] # rough format (arrayN[labels, file ID])
    with open(labels_path / csv_file_name, mode = 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None) # skip headers
        for lines in csv_reader:
            if lines[0] in examples_keys:
                out.append([[lines[1], lines[2], lines[3]], lines[0]])
    
    out = {id: labels for labels, id in out} # format {key: ID | value: labels}
    return out

# Get numpy matrices and labels as training and validation data
def load2(npy_path, labels_path, image_height, image_width, batch_size):
    # Load npy files and convert to single dict
    examples = npy_to_matrix(npy_path, image_height, image_width)

    # Load labels from .csv file
    labels = csv_labels(labels_path, examples.keys())

    # Sort examples and labels by ID number
    sorted_examples = {k: v for k, v in sorted(examples.items(), key=lambda item: item[0])}
    sorted_labels = {k: v for k, v in sorted(labels.items(), key=lambda item: item[0])}

    # Group numpy matrices and labels via tensor slices, then split into training
    # and validation sets
    np_examples = tf.stack(list(sorted_examples.values()))
    np_labels = np.array(list(sorted_labels.values()))
    
    dataset = tf.data.Dataset.from_tensor_slices((np_examples, np_labels))

    dataset = dataset.shuffle(seed=448, buffer_size=1000, reshuffle_each_iteration=False)
    train_size = int(0.7 * len(np_examples))
    validation_size = int(0.3 * len(np_examples))

    train_set = dataset.take(train_size)
    validation_set = dataset.skip(train_size).take(validation_size)

    # Tensorflow magic to cache and prefetch this data, increase performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.batch(batch_size).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_set = validation_set.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    # convert pixels from [0 255] to [0 1], easier on neural net
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_set = train_set.map(lambda x, y: (normalization_layer(x), y))

    return normalized_set, validation_set

# Get images and labels as training and validation data
def load(data_path, image_height, image_width, batch_size):
    # From the folder data_path get images from subfolders (labels = subfolder name)
    train_set = tf.keras.utils.image_dataset_from_directory(
                data_path,
                validation_split=0.3,
                subset="training",
                seed=448,
                image_size=(image_height, image_width),
                batch_size=batch_size
                )
    
    validation_set = tf.keras.utils.image_dataset_from_directory(
                data_path,
                validation_split=0.3,
                subset="validation",
                seed=448,
                image_size=(image_height, image_width),
                #batch_size=batch_size
                )
    
    # Tensorflow magic to cache and prefetch this data, increase performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

    # convert pixels from [0 255] to [0 1], easier on neural net
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_set = train_set.map(lambda x, y: (normalization_layer(x), y))

    return normalized_set, validation_set

# Convolutional Neural Net
def build(input_shape, num_classes):
    model = models.Sequential()

    # Slightly randomizes the input to prevent overfitting and increase accuracy
    model.add(layers.Input(shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomZoom(0.2))

    # Layers of the CNN model
    model.add(layers.Conv2D(16, 5, activation='relu', input_shape=input_shape, padding='valid'))
    model.add(layers.MaxPooling2D(2, strides=2))
    model.add(layers.Conv2D(32, 5, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(72, 3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Take the aformentioned layers of the model and compile together
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy", tf.keras.metrics.SparseCategoricalCrossentropy()],)

    model.summary()

    return model

