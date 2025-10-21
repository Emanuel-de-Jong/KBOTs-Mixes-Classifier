import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import global_params as g
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from sklearn.utils import class_weight
from keras import layers, regularizers
from keras.optimizers import Adam
from Mert import Mert

LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def create_model(layer_array):
    layer_array.insert(0, layers.Input(shape=(Mert.TIME_STEPS, 1024, 25)))
    layer_array.append(layers.Dense(g.label_count, activation='softmax'))
    return Sequential(layer_array)

def calc_class_weight(y_train, should_smooth=False):
    y = np.argmax(y_train, axis=1)
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y)
    
    weights = dict(enumerate(cw))
    if should_smooth:
        weights = smooth_weights(weights)
    
    return weights

def smooth_weights(weights, max_ratio=1.2):
    weights_array = np.array(list(weights.values()))
    smoothed = np.clip(weights_array, 1/max_ratio, max_ratio)
    return dict(zip(weights.keys(), smoothed))

# 64 labels | 6 time steps | 25 songs | global scaler | 250 oversample 0 compensate
# 2025-10-21 14:11 Training took 1687.92 seconds or 28.13 minutes.
# 2025-10-21 14:11 Training Accuracy: 0.6085 | Loss: 1.3586
# 2025-10-21 14:11 Validation Accuracy: 0.3201 | Loss: 2.6769
# 2025-10-21 14:11 Test Accuracy: 0.2951 | Loss: 2.6142
#                   accuracy                           0.30      1098
#                  macro avg       0.31      0.30      0.28      1098
#               weighted avg       0.31      0.30      0.28      1098

# 64 labels | 6 time steps | 25 songs | raw | 250 oversample 0 compensate
# 2025-10-21 12:46 Training took 883.69 seconds or 14.73 minutes.
# 2025-10-21 12:46 Training Accuracy: 0.5816 | Loss: 1.4465
# 2025-10-21 12:46 Validation Accuracy: 0.2882 | Loss: 2.8663
# 2025-10-21 12:46 Test Accuracy: 0.2787 | Loss: 2.5558
#                   accuracy                           0.28      1098
#                  macro avg       0.26      0.27      0.24      1098
#               weighted avg       0.25      0.28      0.24      1098

# 64 labels | 6 time steps | 25 songs | 250 oversample 0 compensate
# 2025-10-21 11:55 Training took 1282.27 seconds or 21.37 minutes.
# 2025-10-21 11:55 Training Accuracy: 0.6341 | Loss: 1.2145
# 2025-10-21 11:55 Validation Accuracy: 0.3208 | Loss: 2.8925
# 2025-10-21 11:55 Test Accuracy: 0.3597 | Loss: 2.4498
#                   accuracy                           0.36      1098
#                  macro avg       0.33      0.37      0.33      1098
#               weighted avg       0.33      0.36      0.33      1098

# 64 labels | 6 time steps | 25 songs | 250 oversample 0.1 compensate
# 2025-10-21 11:08 Training took 1170.87 seconds or 19.51 minutes.
# 2025-10-21 11:08 Training Accuracy: 0.5746 | Loss: 1.4196
# 2025-10-21 11:08 Validation Accuracy: 0.3193 | Loss: 2.6825
# 2025-10-21 11:08 Test Accuracy: 0.3069 | Loss: 2.4371
#                   accuracy                           0.31      1098
#                  macro avg       0.34      0.31      0.28      1098
#               weighted avg       0.34      0.31      0.28      1098

# 64 labels | 6 time steps | 25 songs | 150 undersample
# 2025-10-21 11:32 Training took 827.69 seconds or 13.79 minutes.
# 2025-10-21 11:32 Training Accuracy: 0.5649 | Loss: 1.4311
# 2025-10-21 11:32 Validation Accuracy: 0.2962 | Loss: 2.8039
# 2025-10-21 11:32 Test Accuracy: 0.2923 | Loss: 2.5097
#                   accuracy                           0.29      1098
#                  macro avg       0.29      0.29      0.26      1098
#               weighted avg       0.29      0.29      0.26      1098

# 64 labels | 6 time steps | 25 songs | 150 undersample
# 2025-10-21 00:50 Training took 944.85 seconds or 15.75 minutes.
# 2025-10-21 00:50 Training Accuracy: 0.5968 | Loss: 1.3201
# 2025-10-21 00:50 Validation Accuracy: 0.3174 | Loss: 2.7109
# 2025-10-21 00:50 Test Accuracy: 0.3488 | Loss: 2.4375
#                   accuracy                           0.35      1098
#                  macro avg       0.35      0.35      0.32      1098
#               weighted avg       0.35      0.35      0.32      1098
def m16(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 150 undersample | 0.2 validation
# 2025-10-21 01:42 Training took 838.46 seconds or 13.97 minutes.
# 2025-10-21 01:42 Training Accuracy: 0.5059 | Loss: 1.6198
# 2025-10-21 01:42 Validation Accuracy: 0.3099 | Loss: 2.7277
# 2025-10-21 01:42 Test Accuracy: 0.3106 | Loss: 2.4118
#                   accuracy                           0.31      1098
#                  macro avg       0.29      0.31      0.28      1098
#               weighted avg       0.29      0.31      0.28      1098
def m20(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.4),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.5),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 150 undersample | 0.2 validation
# 2025-10-21 01:28 Training took 478.87 seconds or 7.98 minutes.
# 2025-10-21 01:28 Training Accuracy: 0.6311 | Loss: 1.2012
# 2025-10-21 01:28 Validation Accuracy: 0.2948 | Loss: 3.1869
# 2025-10-21 01:28 Test Accuracy: 0.2404 | Loss: 2.6858
#                   accuracy                           0.24      1098
#                  macro avg       0.21      0.24      0.20      1098
#               weighted avg       0.22      0.24      0.21      1098
def m19(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.1),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 150 undersample | 0.2 validation
# 2025-10-21 01:20 Training took 879.66 seconds or 14.66 minutes.
# 2025-10-21 01:20 Training Accuracy: 0.5087 | Loss: 1.6354
# 2025-10-21 01:20 Validation Accuracy: 0.2939 | Loss: 2.8393
# 2025-10-21 01:20 Test Accuracy: 0.2914 | Loss: 2.5775
#                   accuracy                           0.29      1098
#                  macro avg       0.28      0.29      0.26      1098
#               weighted avg       0.28      0.29      0.26      1098
def m18(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 150 undersample | 0.2 validation
# 2025-10-21 01:05 Training took 897.48 seconds or 14.96 minutes.
# 2025-10-21 01:05 Training Accuracy: 0.4078 | Loss: 1.9733
# 2025-10-21 01:05 Validation Accuracy: 0.2167 | Loss: 2.9638
# 2025-10-21 01:05 Test Accuracy: 0.2541 | Loss: 2.7601
#                   accuracy                           0.25      1098
#                  macro avg       0.22      0.26      0.21      1098
#               weighted avg       0.23      0.25      0.21      1098
def m17(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 200 undersample | 0.2 validation
# 2025-10-20 23:56 Training took 907.91 seconds or 15.13 minutes.
# 2025-10-20 23:56 Training Accuracy: 0.5458 | Loss: 1.4426
# 2025-10-20 23:56 Validation Accuracy: 0.2877 | Loss: 2.6962
# 2025-10-20 23:56 Test Accuracy: 0.2659 | Loss: 2.4946
#                   accuracy                           0.27      1098
#                  macro avg       0.23      0.28      0.23      1098
#               weighted avg       0.23      0.27      0.22      1098
# class_weight is big no
def m15(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        class_weight=calc_class_weight(y_train, should_smooth=True),
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 200 undersample | 0.2 validation
# 2025-10-20 23:34 Training took 676.99 seconds or 11.28 minutes.
# 2025-10-20 23:34 Training Accuracy: 0.6000 | Loss: 1.3140
# 2025-10-20 23:34 Validation Accuracy: 0.3380 | Loss: 2.5731
# 2025-10-20 23:34 Test Accuracy: 0.3242 | Loss: 2.3743
#                   accuracy                           0.32      1098
#                  macro avg       0.32      0.33      0.30      1098
#               weighted avg       0.32      0.32      0.30      1098
# Results exactly the same as m13
def m14(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=5000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 200 undersample | 0.2 validation
# 2025-10-20 23:04 Training took 576.59 seconds or 9.61 minutes.
# 2025-10-20 23:04 Training Accuracy: 0.5746 | Loss: 1.4013
# 2025-10-20 23:04 Validation Accuracy: 0.3399 | Loss: 2.5301
# 2025-10-20 23:04 Test Accuracy: 0.3242 | Loss: 2.3743
#                   accuracy                           0.32      1098
#                  macro avg       0.32      0.33      0.30      1098
#               weighted avg       0.32      0.32      0.30      1098
# Deep models NEED GlobalAveragePooling2D instead of Flatten
def m13(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.0001)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((1,4)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_regularizer=kernel_regularizer),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )

    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=1000,
        validation_data=validation_data,
        callbacks=[reduce_lr, early_stopping],
    )

    return model, training_data

# 64 labels | 6 time steps | 25 songs | 250 undersample | song split
# 2025-10-20 21:55 Training took 329.21 seconds or 5.49 minutes.
# 2025-10-20 21:55 Training Accuracy: 0.9180 | Loss: 0.5810
# 2025-10-20 21:55 Validation Accuracy: 0.3424 | Loss: 4.0613
# 2025-10-20 21:55 Test Accuracy: 0.2823 | Loss: 2.9738
#                   accuracy                           0.28      1098
#                  macro avg       0.25      0.28      0.25      1098
#               weighted avg       0.25      0.28      0.25      1098
# BatchNormalization was what broke everything...
def m12(name, X_train, y_train, validation_data):
    kernel_regularizer = regularizers.l2(0.01)
    model = create_model([
        layers.Conv2D(64, (5,5), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.SpatialDropout2D(0.1),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.SpatialDropout2D(0.2),

        layers.Flatten(),

        layers.Dense(128, activation='relu', kernel_regularizer=kernel_regularizer),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=LOSS,
        metrics=METRICS,
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

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

# 64 labels | 6 time steps | 25 songs | 250 undersample | song split
# 2025-10-20 21:01 Training took 905.90 seconds or 15.10 minutes.
# 2025-10-20 21:01 Training Accuracy: 0.0124 | Loss: 4.1589
# 2025-10-20 21:01 Validation Accuracy: 0.0156 | Loss: 4.1589
# 2025-10-20 21:01 Test Accuracy: 0.0164 | Loss: 4.1588
#                   accuracy                           0.02      1098
#                  macro avg       0.00      0.02      0.00      1098
#               weighted avg       0.00      0.02      0.00      1098
# Extreme overfitting

# 64 labels | 5 time steps | 25 songs | none
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
# 2025-10-20 12:44 Validation Accuracy: 0.5073 | Loss: 2.3055
# 2025-10-20 12:44 Test Accuracy: 0.2268 | Loss: 3.6503
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
