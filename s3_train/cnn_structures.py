import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import s0_utils.global_params as g
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from sklearn.utils import class_weight
from keras import layers, regularizers
from keras.optimizers import Adam
from s0_utils.Mert import Mert

LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def create_model(layer_array):
    layer_array.insert(0, layers.Input(shape=g.DATA_SHAPE))
    layer_array.append(layers.Dense(g.LABEL_COUNT, activation='softmax'))
    return Sequential(layer_array)

def fit_model(model, train_seq, validate_seq, callbacks):
    training_data = None
    if g.USE_SHARDS_IN_TRAINING:
        training_data = model.fit(
            train_seq,
            batch_size=g.MODEL_BATCH_SIZE,
            epochs=5000,
            validation_data=validate_seq,
            callbacks=callbacks
        )
    else:
        training_data = model.fit(
            train_seq[0],
            train_seq[1],
            batch_size=g.MODEL_BATCH_SIZE,
            epochs=5000,
            validation_data=validate_seq,
            callbacks=callbacks
        )
    
    return training_data

def m16(name, train_seq, validate_seq):
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
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.25)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=18, restore_best_weights=True)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    training_data = fit_model(model, train_seq, validate_seq, [reduce_lr, early_stopping])

    return model, training_data

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
