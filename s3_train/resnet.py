import os

os.environ["KERAS_BACKEND"] = "torch"

import s0_utils.global_params as g
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, Model
from keras import layers, regularizers
from keras.optimizers import Adam

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

def residual_block(input_tensor, filter_count, stride=(1,1), kernel_regularizer=None):
    shortcut_tensor = input_tensor

    output_tensor = layers.Conv2D(
        filter_count,
        (3,3),
        strides=stride,
        padding='same',
        activation=None,
        kernel_regularizer=kernel_regularizer
    )(input_tensor)
    output_tensor = layers.BatchNormalization()(output_tensor)
    output_tensor = layers.Activation('relu')(output_tensor)

    output_tensor = layers.Conv2D(
        filter_count,
        (3,3),
        strides=(1,1),
        padding='same',
        activation=None,
        kernel_regularizer=kernel_regularizer
    )(output_tensor)
    output_tensor = layers.BatchNormalization()(output_tensor)

    if shortcut_tensor.shape[-1] != filter_count or stride != (1,1):
        shortcut_tensor = layers.Conv2D(
            filter_count,
            (1,1),
            strides=stride,
            padding='same',
            activation=None,
            kernel_regularizer=kernel_regularizer
        )(shortcut_tensor)
        shortcut_tensor = layers.BatchNormalization()(shortcut_tensor)

    output_tensor = layers.Add()([output_tensor, shortcut_tensor])
    output_tensor = layers.Activation('relu')(output_tensor)

    return output_tensor

def m16(name, train_seq, validate_seq):
    kernel_regularizer = regularizers.l2(0.0001)

    input_tensor = layers.Input(shape=g.DATA_SHAPE)

    output_tensor = layers.Conv2D(
        64,
        (5,5),
        padding='same',
        activation=None,
        kernel_regularizer=kernel_regularizer
    )(input_tensor)
    output_tensor = layers.BatchNormalization()(output_tensor)
    output_tensor = layers.Activation('relu')(output_tensor)
    output_tensor = layers.MaxPooling2D((1,4))(output_tensor)
    output_tensor = layers.SpatialDropout2D(0.3)(output_tensor)

    output_tensor = residual_block(
        output_tensor,
        128,
        stride=(1,2),
        kernel_regularizer=kernel_regularizer
    )
    output_tensor = residual_block(
        output_tensor,
        128,
        stride=(1,1),
        kernel_regularizer=kernel_regularizer
    )
    output_tensor = layers.MaxPooling2D((1,2))(output_tensor)
    output_tensor = layers.SpatialDropout2D(0.3)(output_tensor)

    output_tensor = residual_block(
        output_tensor,
        256,
        stride=(1,2),
        kernel_regularizer=kernel_regularizer
    )
    output_tensor = residual_block(
        output_tensor,
        256,
        stride=(1,1),
        kernel_regularizer=kernel_regularizer
    )
    output_tensor = layers.MaxPooling2D((1,2))(output_tensor)
    output_tensor = layers.SpatialDropout2D(0.3)(output_tensor)

    output_tensor = layers.GlobalAveragePooling2D()(output_tensor)

    output_tensor = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=kernel_regularizer
    )(output_tensor)

    output_tensor = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=kernel_regularizer
    )(output_tensor)

    output_tensor = layers.Dense(
        g.LABEL_COUNT,
        activation='softmax'
    )(output_tensor)

    model = Model(input_tensor, output_tensor)

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
