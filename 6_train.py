import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import cnn_structures as cnns
import numpy as np
import joblib
import random
import torch
import json
import time
import gc
import global_params as g
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from DiskShardedSequence import DiskShardedSequence
from keras.models import load_model
from Utils import Logger

model, history = None, None

logger = Logger(g.MODELS_DIR / "train.log")

data_paths = g.get_all_data_paths(5)

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
    model_path = g.CACHE_DIR / f'model_{g.NAME}.keras'
    history_path = g.MODELS_DIR / f'history.json'
    if not os.path.exists(model_path) or not os.path.exists(history_path):
        return None, None
    
    model = load_model(model_path)
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, history
    
def save_model(name, model, training_data):
    model.save(g.CACHE_DIR / f'model_{name}.keras')

    history = {}
    history["accuracy"] = training_data.history["accuracy"]
    history["val_accuracy"] = training_data.history["val_accuracy"]
    history["loss"] = training_data.history["loss"]
    history["val_loss"] = training_data.history["val_loss"]

    with open(g.MODELS_DIR / f'history_{name}.json', 'w') as f:
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

    plt.savefig(g.MODELS_DIR / f'test_acc_loss_{name}.png', bbox_inches='tight')
    plt.close()

def test(model, history, name=""):
    draw_acc_and_loss_graphs(history, name)

    logger.writeln(f"Training Accuracy: {history['accuracy'][-1]:.4f} | Loss: {history['loss'][-1]:.4f}")
    logger.writeln(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f} | Loss: {history['val_loss'][-1]:.4f}")

    y_true = []
    y_pred_sk = []

    for data_path in data_paths[g.DataSetType.test]:
        df = joblib.load(data_path)
        X_test = np.stack(df["data"].to_numpy())
        y_test = df["label"].to_numpy()

        y_pred = model.predict(X_test)
        y_pred_sk.extend(np.argmax(y_pred, axis=-1))
        y_true.extend(y_test)

        del df, X_test, y_pred
        gc.collect()

    report = classification_report(y_true, y_pred_sk, target_names = g.LABELS)
    logger.writeln(report)

    cm = confusion_matrix(y_true, y_pred_sk)
    disp = ConfusionMatrixDisplay(cm, display_labels = g.LABELS)

    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.tight_layout(pad=3.0)
    plt.savefig(g.MODELS_DIR / f'test_matrix_{name}.png', bbox_inches='tight')
    plt.close()

    test_loss, test_accuracy = model.evaluate(
        DiskShardedSequence(data_paths[g.DataSetType.test], shuffle=False),
        verbose=0
    )
    logger.writeln(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

def train(model_func):
    name = g.NAME
    # name = model_func.__name__
    logger.writeln(name)

    train_seq = DiskShardedSequence(
        data_paths[g.DataSetType.train],
        shuffle=True
    )

    validate_seq = DiskShardedSequence(
        data_paths[g.DataSetType.validate],
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
    # train(cnns.m1)
    # train(cnns.m2)
    # train(cnns.m3)
    # train(cnns.m4)
    # train(cnns.m5)
    # train(cnns.m6)
    # train(cnns.m7)
    # train(cnns.m8)
    # train(cnns.m9)
    # train(cnns.m10)
    # train(cnns.m11)
    # train(cnns.m12)
    # train(cnns.m13)
    # train(cnns.m14)
    # train(cnns.m15)
    train(cnns.m16)
    # train(cnns.m17)
    # train(cnns.m18)
    # train(cnns.m19)
    # train(cnns.m20)

    # train(cnns.m21)
    # train(cnns.m22)
    # train(cnns.m23)
    # train(cnns.m24)
    # train(cnns.m25)
    # train(cnns.m26)
    # train(cnns.m27)
    # train(cnns.m28)
    # train(cnns.m29)
    # train(cnns.m30)

else:
    test(model, history)
