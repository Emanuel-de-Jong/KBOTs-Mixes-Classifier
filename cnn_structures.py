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

# Epoch 151/200
# Training took 449.15 seconds or 7.49 minutes.
# Training Accuracy: 0.7236 | Loss: 0.9025
# Validation Accuracy: 0.7849 | Loss: 0.8039
# Test Accuracy: 0.6429 | Loss: 1.3069
#               precision    recall  f1-score   support
#     accuracy                           0.64        28
#    macro avg       0.50      0.67      0.57        28
# weighted avg       0.49      0.64      0.55        28
def m25(X_train, y_train, validation_data):
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
