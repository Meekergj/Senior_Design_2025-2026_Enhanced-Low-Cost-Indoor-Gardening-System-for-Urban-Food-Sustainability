# Copyright info and other stuff
#
#

import tensorflow as tf

from keras import datasets, layers, models, metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

csv_labels_filename = 'Image Label and IDs.csv'

# Helper method, places npy files into single (resized) matrix
def npy_to_matrix(npy_path, image_height, image_width):
    out = [] # rough format (matrixN[file info, file ID], ...)
    for file in os.scandir(npy_path):
        loaded_np = np.load(npy_path / file.name)
        out.append([tf.keras.preprocessing.image.smart_resize(loaded_np, (image_height, image_width)), str(file.name.split(os.extsep)[0])])
    out = {id: matrix for matrix, id in out} # format {key: ID | value: image matrix}
    return out

# Helper method, gets IDs/labels from csv file and places into dictionary (exclues ids not in examples)
def csv_labels(labels_path, examples_keys, csv_file_name):
    out = [] # rough format (arrayN[labels, file ID])
    with open(labels_path / csv_file_name, mode = 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None) # skip headers
        for lines in csv_reader:
            if lines[0] in examples_keys:
                out.append([[float(lines[1]), float(lines[2]), float(lines[3])], lines[0]])
    
    out = {id: labels for labels, id in out} # format {key: ID | value: labels}
    return out

# Get numpy matrices and labels as training and validation data
def load(npy_path, labels_path, image_height, image_width, batch_size):
    # Load npy files and convert to single dict
    examples = npy_to_matrix(npy_path, image_height, image_width)

    # Load labels from .csv file
    labels = csv_labels(labels_path, examples.keys(), csv_labels_filename)

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
    train_set = train_set.shuffle(1000).batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    validation_set = validation_set.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    # convert pixels from [0 255] to [0 1], easier on neural net
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_set = train_set.map(lambda x, y: (normalization_layer(x), y))

    return normalized_set, validation_set

# Convolutional Neural Net
def build(input_shape, num_labels):
    model_input = layers.Input(shape=input_shape)

    # Slightly randomizes the input to prevent overfitting and increase accuracy
    x = layers.RandomFlip(mode="horizontal_and_vertical")(model_input)
    x = layers.RandomRotation(factor=0.2)(x)
    x = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(x)

    # Layers of the CNN model
    x = layers.Conv2D(filters=16, kernel_size=5, activation='relu', input_shape=input_shape, padding='valid')(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    #x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(filters=72, kernel_size=3, activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # Outputs (predicted value of each label)
    model_output = layers.Dense(num_labels, activation='sigmoid')(x)

    # Build the model
    model = models.Model(inputs=model_input, outputs=model_output)

    # Take the model and compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy", "mse"],)

    model.summary()

    return model

# Convolutional Neural Net using 2D depthwise convolution
def build_depthwise(input_shape, num_labels):
    model_input = layers.Input(shape=input_shape)

    # Slightly randomizes the input to prevent overfitting and increase accuracy
    x = layers.RandomFlip("horizontal")(model_input)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)

    # Layers of the CNN model
    x = layers.DepthwiseConv2D(depth_multiplier=8, kernel_size=5, activation='relu', input_shape=input_shape, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.DepthwiseConv2D(depth_multiplier=16, kernel_size=5, activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.DepthwiseConv2D(depth_multiplier=16, kernel_size=3, activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.DepthwiseConv2D(depth_multiplier=16, kernel_size=3, activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # Outputs (predicted value of each label)
    model_output = layers.Dense(num_labels, activation='sigmoid')(x)

    # Build the model
    model = models.Model(inputs=model_input, outputs=model_output)

    # Take the model and compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy", "mse"],)

    model.summary()

    return model