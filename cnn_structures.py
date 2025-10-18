import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import joblib
import json
import time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras import layers, regularizers
from keras.utils import to_categorical
from keras.optimizers import Adam
from pathlib import Path
from Utils import Logger
from Mert import Mert

cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))

label_count = len(labels)

LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def create_model(layer_array):
    layer_array.insert(0, layers.Input(shape=(Mert.TIME_STEPS, 1024, 25)))
    layer_array.append(layers.Dense(label_count, activation='softmax'))
    return Sequential(layer_array)

def m1(name, X_train, y_train, validation_data):
    model = create_model([
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=15,
        validation_data=validation_data,
    )

    return model, training_data

def m2(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

def m3(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.DepthwiseConv2D((1,5), padding='same', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),

        layers.SeparableConv2D(64, (1,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((1,2)),

        layers.SeparableConv2D(128, (1,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((1,2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# Training took 436.24 seconds or 7.27 minutes.
# Test Accuracy: 0.5000 | Loss: 1.7629
# Training Accuracy: 0.4986 | Loss: 1.6503
# Validation Accuracy: 0.4946 | Loss: 1.5341
#               precision    recall  f1-score   support
#     accuracy                           0.50        28
#    macro avg       0.49      0.53      0.49        28
# weighted avg       0.45      0.50      0.45        28
def m4(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.3),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.Dropout(0.3),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
        # class_weight='balanced',
    )

    return model, training_data

# Epoch 151/200
# Training took 449.15 seconds or 7.49 minutes.
# Training Accuracy: 0.7236 | Loss: 0.9025
# Validation Accuracy: 0.7849 | Loss: 0.8039
# Test Accuracy: 0.6429 | Loss: 1.3069
#               precision    recall  f1-score   support
#     accuracy                           0.64        28
#    macro avg       0.50      0.67      0.57        28
# weighted avg       0.49      0.64      0.55        28
def m5(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.3),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.Dropout(0.3),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
        # class_weight='balanced',
    )

    return model, training_data

def m6(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.DepthwiseConv2D((5,1), padding='same', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.SeparableConv2D(64, (5,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2,2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

def m7(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.Dropout(0.2),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

def m8(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (7, 7), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.SpatialDropout2D(0.2),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data
