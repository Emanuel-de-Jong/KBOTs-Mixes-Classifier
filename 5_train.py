import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import cnn_structures as cnns
import numpy as np
import random
import torch
import json
import time
import global_params as g
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from keras.models import load_model
from keras.utils import to_categorical
from Utils import Logger

g.load_data(4)

model, history = None, None

logger = Logger(g.MODELS_DIR / "train.log")

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# set_seed()

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
validate_data = g.data[g.data["data_set"] == g.DataSetType.validate]
test_data = g.data[g.data["data_set"] == g.DataSetType.test]

X_train = np.stack(train_data["data"].to_numpy())
X_validate = np.stack(validate_data["data"].to_numpy())
X_test = np.stack(test_data["data"].to_numpy())

y_train = train_data["label"].to_numpy()
y_validate = validate_data["label"].to_numpy()
y_test = test_data["label"].to_numpy()

y_train_hot = to_categorical(y_train)
y_validate_hot = to_categorical(y_validate)
y_test_hot = to_categorical(y_test)

validation_data=(X_validate, y_validate_hot)

def load_existing_model():
    model_path = g.CACHE_DIR / f'model_global.keras'
    history_path = g.MODELS_DIR / f'history.json'
    if not os.path.exists(model_path) or not os.path.exists(history_path):
        return None, None
    
    model = load_model(model_path)
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, history
    
def save_model(name, model, training_data):
    # model.save(g.CACHE_DIR / f'model_global.keras')
    model.save(g.MODELS_DIR / f'model_{name}.keras')

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

    y_pred = model.predict(X_test)
    y_pred_sk = np.argmax(y_pred, axis=-1)

    report = classification_report(y_test, y_pred_sk, target_names = g.labels)
    logger.writeln(report)

    cm = confusion_matrix(y_test, y_pred_sk)
    disp = ConfusionMatrixDisplay(cm, display_labels = g.labels)

    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.tight_layout(pad=3.0)
    plt.savefig(g.MODELS_DIR / f'test_matrix_{name}.png', bbox_inches='tight')
    plt.close()

    test_loss, test_accuracy = model.evaluate(X_test, y_test_hot, verbose=0)
    logger.writeln(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

def train(model_func):
    name = model_func.__name__
    logger.writeln(name)

    start_time = time.time()
    model, training_data = model_func(name, X_train, y_train_hot, validation_data)
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
    train(cnns.m14)
    # train(cnns.m15)
    # train(cnns.m16)
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
