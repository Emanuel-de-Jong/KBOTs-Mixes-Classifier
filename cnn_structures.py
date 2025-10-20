import os

os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import numpy as np
import joblib
import global_params as g
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from sklearn.utils import class_weight
from keras import layers, regularizers
from keras.optimizers import Adam
from pathlib import Path
from Mert import Mert

LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def create_model(layer_array):
    layer_array.insert(0, layers.Input(shape=(Mert.TIME_STEPS, 1024, 25)))
    layer_array.append(layers.Dense(g.label_count, activation='softmax'))
    return Sequential(layer_array)

def calc_class_weight(y_train):
    y = np.argmax(y_train, axis=1)
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y)
    return dict(enumerate(cw))

# 64 labels | 5 time steps | 25 songs | raw
# 2025-10-20 13:34 Training took 1775.68 seconds or 29.59 minutes.
# 2025-10-20 13:34 Training Accuracy: 0.7784 | Loss: 0.8532
# 2025-10-20 13:34 Validation Accuracy: 0.5502 | Loss: 2.0031
# 2025-10-20 13:34 Test Accuracy: 0.2659 | Loss: 4.1750
#                   accuracy                           0.27      1098
#                  macro avg       0.27      0.26      0.25      1098
#               weighted avg       0.27      0.27      0.25      1098
def m11(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.01)
    model = create_model([
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),

        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Flatten(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=500,
        validation_data=validation_data,
        class_weight=calc_class_weight(y_train),
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 5 time steps | 25 songs | -1 undersample
# 2025-10-20 12:44 Training took 513.40 seconds or 8.56 minutes.
# 2025-10-20 12:44 Training Accuracy: 0.7873 | Loss: 1.0380
# 2025-10-20 12:44 Test Accuracy: 0.2268 | Loss: 3.6503
# 2025-10-20 12:44 Validation Accuracy: 0.5073 | Loss: 2.3055
#                   accuracy                           0.23      1098
#                  macro avg       0.24      0.23      0.23      1098
#               weighted avg       0.24      0.23      0.23      1098
def m10(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.01)
    model = create_model([
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),

        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Flatten(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=500,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 5 time steps | 25 songs | 175 undersample
# 2025-10-19 11:20 Training took 110.54 seconds or 1.84 minutes.
# 2025-10-19 11:20 Training Accuracy: 0.9044 | Loss: 0.2991
# 2025-10-19 11:20 Validation Accuracy: 0.5401 | Loss: 2.4056
# 2025-10-19 11:20 Test Accuracy: 0.2477 | Loss: 4.8794
#                   accuracy                           0.25      1098
#                  macro avg       0.25      0.25      0.23      1098
#               weighted avg       0.25      0.25      0.24      1098
def m9(name, X_train, y_train, validation_data):
    model = create_model([
        layers.MaxPooling2D((1, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((1, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((1, 2)),

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





# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 22:59 Training took 31.22 seconds or 0.52 minutes.
# 2025-10-18 22:59 Training Accuracy: 0.9973 | Loss: 0.0155
# 2025-10-18 22:59 Validation Accuracy: 0.9140 | Loss: 0.2196
# 2025-10-18 22:59 Test Accuracy: 0.8571 | Loss: 0.2975
# 2025-10-18 22:59               precision    recall  f1-score   support
#     accuracy                           0.86        28
#    macro avg       0.89      0.87      0.85        28
# weighted avg       0.91      0.86      0.86        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:09 Training took 436.65 seconds or 7.28 minutes.
# 2025-10-18 23:09 Training Accuracy: 0.6640 | Loss: 0.9834
# 2025-10-18 23:09 Validation Accuracy: 0.6559 | Loss: 1.0881
# 2025-10-18 23:09 Test Accuracy: 0.6429 | Loss: 1.0346
# 2025-10-18 23:09               precision    recall  f1-score   support
#     accuracy                           0.64        28
#    macro avg       0.69      0.67      0.63        28
# weighted avg       0.68      0.64      0.61        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:23 Training took 497.57 seconds or 8.29 minutes.
# 2025-10-18 23:23 Training Accuracy: 0.7696 | Loss: 0.7818
# 2025-10-18 23:23 Validation Accuracy: 0.8602 | Loss: 0.6210
# 2025-10-18 23:23 Test Accuracy: 0.6071 | Loss: 1.4300
# 2025-10-18 23:23               precision    recall  f1-score   support
#     accuracy                           0.61        28
#    macro avg       0.50      0.63      0.55        28
# weighted avg       0.48      0.61      0.52        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:33 Training took 307.64 seconds or 5.13 minutes.
# 2025-10-18 23:33 Training Accuracy: 0.7913 | Loss: 0.7716
# 2025-10-18 23:33 Validation Accuracy: 0.7312 | Loss: 0.9193
# 2025-10-18 23:33 Test Accuracy: 0.5714 | Loss: 1.6012
# 2025-10-18 23:33               precision    recall  f1-score   support
#     accuracy                           0.57        28
#    macro avg       0.63      0.58      0.56        28
# weighted avg       0.64      0.57      0.57        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:28 Training took 305.02 seconds or 5.08 minutes.
# 2025-10-18 23:28 Training Accuracy: 0.8862 | Loss: 0.4473
# 2025-10-18 23:28 Validation Accuracy: 0.8817 | Loss: 0.5150
# 2025-10-18 23:28 Test Accuracy: 0.5357 | Loss: 1.4849
# 2025-10-18 23:28               precision    recall  f1-score   support
#     accuracy                           0.54        28
#    macro avg       0.43      0.57      0.47        28
# weighted avg       0.41      0.54      0.45        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:02 Training took 172.65 seconds or 2.88 minutes.
# 2025-10-18 23:02 Training Accuracy: 0.5827 | Loss: 1.0414
# 2025-10-18 23:02 Validation Accuracy: 0.4946 | Loss: 1.2664
# 2025-10-18 23:02 Test Accuracy: 0.4643 | Loss: 1.5186
# 2025-10-18 23:02               precision    recall  f1-score   support
#     accuracy                           0.46        28
#    macro avg       0.36      0.50      0.37        28
# weighted avg       0.37      0.46      0.37        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:14 Training took 302.44 seconds or 5.04 minutes.
# 2025-10-18 23:14 Training Accuracy: 0.4824 | Loss: 1.7260
# 2025-10-18 23:14 Validation Accuracy: 0.4516 | Loss: 1.7018
# 2025-10-18 23:14 Test Accuracy: 0.3929 | Loss: 1.8909
# 2025-10-18 23:14               precision    recall  f1-score   support
#     accuracy                           0.39        28
#    macro avg       0.32      0.43      0.35        28
# weighted avg       0.27      0.39      0.31        28
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

# 6 labels | 90 time steps | 10 songs | -1 undersample
# 2025-10-18 23:41 Training took 462.21 seconds or 7.70 minutes.
# 2025-10-18 23:41 Training Accuracy: 0.1680 | Loss: 1.7923
# 2025-10-18 23:41 Validation Accuracy: 0.1613 | Loss: 1.7927
# 2025-10-18 23:41 Test Accuracy: 0.1786 | Loss: 1.7919
# 2025-10-18 23:41               precision    recall  f1-score   support
#     accuracy                           0.18        28
#    macro avg       0.03      0.17      0.05        28
# weighted avg       0.03      0.18      0.05        28
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
