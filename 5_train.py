import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import cnn_structures as cnns
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

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))
X = joblib.load(cache_dir / f'X_train.joblib')
X_test = joblib.load(cache_dir / f'X_test.joblib')
y_pre = joblib.load(cache_dir / f'y_train.joblib')
y_test_pre = joblib.load(cache_dir / f'y_test.joblib')

model, history = None, None

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

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.15, stratify=y, random_state=1)

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
    
def save_model(name, model, training_data):
    # model.save(cache_dir / f'model_global.keras')
    model.save(models_dir / f'model_{name}.keras')

    history = {}
    history["accuracy"] = training_data.history["accuracy"]
    history["val_accuracy"] = training_data.history["val_accuracy"]
    history["loss"] = training_data.history["loss"]
    history["val_loss"] = training_data.history["val_loss"]

    with open(models_dir / f'history_{name}.json', 'w') as f:
        json.dump(history, f)
    
    return history

def draw_acc_and_loss_graphs(history, name):
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

    plt.savefig(models_dir / f'test_acc_loss_{name}.png', bbox_inches='tight')
    plt.close()

def test(model, history, name=""):
    draw_acc_and_loss_graphs(history, name)

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
    plt.savefig(models_dir / f'test_matrix_{name}.png', bbox_inches='tight')
    plt.close()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.writeln(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

def train(name, model_func):
    logger.writeln(name)

    start_time = time.time()
    model, training_data = model_func(name, X_train, y_train, validation_data)
    elapsed_time = time.time() - start_time
    logger.writeln(f"Training took {elapsed_time:.2f} seconds or {elapsed_time/60:.2f} minutes.")

    history = save_model(name, model, training_data)

    test(model, history, name)

    return model, history

# model, history = load_existing_model()
if model is None:
    # train("m1", cnns.m1)
    # train("m2", cnns.m2)
    # train("m3", cnns.m3)
    # train("m4", cnns.m4)
    # train("m5", cnns.m5)
    # train("m6", cnns.m6)
    # train("m7", cnns.m7)
    # train("m8", cnns.m8)
    # train("m9", cnns.m9)
    # train("m10", cnns.m10)
    train("m11", cnns.m11)

else:
    test(model, history)
