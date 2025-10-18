import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import joblib
import json
import time
import os
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from pathlib import Path
from Utils import Logger
from Mert import Mert

os.environ["KERAS_BACKEND"] = "torch"

from keras import layers
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))
X = joblib.load(cache_dir / f'X_train.joblib')
X_test = joblib.load(cache_dir / f'X_test.joblib')
y_pre = joblib.load(cache_dir / f'y_train.joblib')
y_test_pre = joblib.load(cache_dir / f'y_test.joblib')

logger = Logger(models_dir / "train.log")

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

label_count = len(labels)

y = to_categorical(y_pre)
y_test = to_categorical(y_test_pre)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

loss = 'categorical_crossentropy'
metrics = ['accuracy']
validation_data=(X_validate, y_validate)

def load_existing_model():
    model_path = cache_dir / f'model_global.keras'
    history_path = models_dir / f'history.json'
    if not os.path.exists(model_path) or not os.path.exists(history_path):
        return None, None
    
    model = load_model(model_path)
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, history
    
def save_model(model, training_data):
    model.save(cache_dir / f'model_global.keras')

    history = {}
    history["accuracy"] = training_data.history["accuracy"]
    history["val_accuracy"] = training_data.history["val_accuracy"]
    history["loss"] = training_data.history["loss"]
    history["val_loss"] = training_data.history["val_loss"]

    with open(models_dir / f'history.json', 'w') as f:
        json.dump(history, f)
    
    return history

def train():
    model = Sequential([
        layers.Input(shape=(Mert.TIME_STEPS, 1024, 25)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(label_count, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics)

    training_data = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=validation_data)

    history = save_model(model, training_data)

    return model, history

# model, history = load_existing_model()
model, history = None, None
if not model:
    start_time = time.time()
    model, history = train()
    elapsed_time = time.time() - start_time
    logger.writeln(f"Training took {elapsed_time:.2f} seconds or {elapsed_time/60:.2f} minutes.")

def draw_acc_and_loss_graphs(history):
    plt.figure(figsize=(9, 2))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.savefig(models_dir / f'test_acc_loss.png', bbox_inches='tight')
    plt.close()

def test(model, history):
    draw_acc_and_loss_graphs(history)

    logger.writeln(f"Training Accuracy: {history['accuracy'][-1]:.4f} | Loss: {history['loss'][-1]:.4f}")
    logger.writeln(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f} | Loss: {history['val_loss'][-1]:.4f}")

    y_pred = model.predict(X_test)
    y_pred_sk = np.argmax(y_pred, axis=-1)

    report = classification_report(y_test_pre, y_pred_sk, target_names = labels)
    logger.writeln(report)

    cm = confusion_matrix(y_test_pre, y_pred_sk)
    disp = ConfusionMatrixDisplay(cm, display_labels = labels)

    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.tight_layout(pad=3.0)
    plt.savefig(models_dir / f'test_matrix.png', bbox_inches='tight')
    plt.close()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.writeln(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

test(model, history)
