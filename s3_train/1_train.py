import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import s3_train.cnn_structures as cnns
import numpy as np
import random
import torch
import json
import time
import gc
import s0_utils.global_params as g
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from s3_train.DiskShardedSequence import DiskShardedSequence
from keras.models import load_model
from s0_utils.Logger import Logger

TRAINING_DIR = Path("s3_train/training")
TRAINING_DIR.mkdir(exist_ok=True)

model, history = None, None

logger = Logger(TRAINING_DIR / "training.log")

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# set_seed()

def load_existing_model():
    model_path = g.MODELS_DIR / f'model_{g.NAME}.keras'
    history_path = TRAINING_DIR / f'history_{g.NAME}.json'
    if not os.path.exists(model_path) or not os.path.exists(history_path):
        return None, None
    
    model = load_model(model_path)
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, history
    
def save_model(name, model, training_data):
    model.save(g.MODELS_DIR / f'model_{name}.keras')

    history = {}
    history["accuracy"] = training_data.history["accuracy"]
    history["val_accuracy"] = training_data.history["val_accuracy"]
    history["loss"] = training_data.history["loss"]
    history["val_loss"] = training_data.history["val_loss"]

    with open(TRAINING_DIR / f'history_{name}.json', 'w') as f:
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

    plt.savefig(TRAINING_DIR/ f'test_acc_loss_{name}.png', bbox_inches='tight')
    plt.close()

def draw_confusion_matrix(y_true, y_pred_sk, name=""):
    cm = confusion_matrix(y_true, y_pred_sk)

    per_label = []
    for i, label in enumerate(g.LABELS):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0.0

        row = cm[i].copy()
        row[i] = 0
        wrong_idx = row.argmax()
        wrong_count = row[wrong_idx]

        wrong_label = g.LABELS[wrong_idx] if wrong_count > 0 else None
        per_label.append((acc, label, wrong_label))

    per_label.sort(key=lambda x: x[0])

    logger.writeln("\nPer-label accuracy (sorted):")
    for acc, label, wrong_label in per_label:
        if wrong_label is not None:
            logger.writeln(f"{acc:.2f} {label} ({wrong_label})")
        else:
            logger.writeln(f"{acc:.2f} {label}")

    disp = ConfusionMatrixDisplay(cm, display_labels=g.LABELS)
    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.tight_layout(pad=3.0)
    plt.savefig(TRAINING_DIR / f'test_matrix_{name}.png', bbox_inches='tight')
    plt.close()

def test(model, history, name=""):
    draw_acc_and_loss_graphs(history, name)

    logger.writeln(f"Training Accuracy: {history['accuracy'][-1]:.4f} | Loss: {history['loss'][-1]:.4f}")
    logger.writeln(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f} | Loss: {history['val_loss'][-1]:.4f}")

    y_true = []
    y_pred_sk = []

    test_data_paths = list(g.iter_zarr_data_paths(6, g.DataSetType.test))
    test_seq = DiskShardedSequence(test_data_paths, shuffle=False)
    for i in range(len(test_seq)):
        X_test, y_test = test_seq[i]
        y_pred = model.predict(X_test)
        y_pred_sk.extend(np.argmax(y_pred, axis=-1))
        y_true.extend(np.argmax(y_test, axis=-1))

        del X_test, y_test, y_pred
        gc.collect()

    report = classification_report(y_true, y_pred_sk, target_names=g.LABELS)
    logger.writeln(report)

    draw_confusion_matrix(y_true, y_pred_sk, name)

    test_loss, test_accuracy = model.evaluate(
        test_seq,
        verbose=0
    )
    logger.writeln(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

def train(model_func):
    name = g.NAME
    # name = model_func.__name__
    logger.writeln(name)

    train_seq = DiskShardedSequence(
        list(g.iter_zarr_data_paths(6, g.DataSetType.train)),
        shuffle=True
    )

    validate_seq = DiskShardedSequence(
        list(g.iter_zarr_data_paths(6, g.DataSetType.validate)),
        shuffle=False
    )

    start_time = time.time()
    model, training_data = model_func(name, train_seq, validate_seq)
    elapsed_time = time.time() - start_time
    logger.writeln(f"Training took {elapsed_time:.2f} seconds or {elapsed_time/60:.2f} minutes.")

    history = save_model(name, model, training_data)

    test(model, history, name)

    return model, history

# model, history = load_existing_model()
if model is None:
    train(cnns.m16)

else:
    test(model, history)
