import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Classifier
import joblib
import time
import sys
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from Utils import Logger
from enum import Enum

class ScalingType(Enum):
    RAW = 0
    SCALE = 1
    NORM = 2

class ClassifierToTrain():
    X_train = None
    X_test = None
    model = None
    accuracy = None

    def __init__(self, name, func, scaling_type):
        self.name = name
        self.func = func
        self.scaling_type = scaling_type

VERBOSE = False
CV = 4
CLASSIFIERS_TO_TRAIN = [
    ClassifierToTrain('KNeighbors', Classifier.train_KNeighbors, ScalingType.NORM),
    ClassifierToTrain('SVC', Classifier.train_SVC, ScalingType.SCALE),
    ClassifierToTrain('GaussianNB', Classifier.train_GaussianNB, ScalingType.SCALE),
    # Takes too long
    # ClassifierToTrain('LogisticRegression', Classifier.train_LogisticRegression, ScalingType.SCALE),
    ClassifierToTrain('MLP', Classifier.train_MLP, ScalingType.SCALE),
    ClassifierToTrain('DecisionTree', Classifier.train_DecisionTree, ScalingType.RAW),
    ClassifierToTrain('RandomForest', Classifier.train_RandomForest, ScalingType.RAW),
    ClassifierToTrain('ExtraTrees', Classifier.train_ExtraTrees, ScalingType.RAW),
    ClassifierToTrain('GradientBoosting', Classifier.train_GradientBoosting, ScalingType.SCALE),
]

# patch_sklearn()
# set_config(target_offload="gpu")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))
X_train = joblib.load(cache_dir / f'X_train.joblib')
X_test = joblib.load(cache_dir / f'X_test.joblib')
X_train_norm = joblib.load(cache_dir / f'X_train_norm.joblib')
X_test_norm = joblib.load(cache_dir / f'X_test_norm.joblib')
X_train_scale = joblib.load(cache_dir / f'X_train_scale.joblib')
X_test_scale = joblib.load(cache_dir / f'X_test_scale.joblib')
y_train = joblib.load(cache_dir / f'y_train.joblib')
y_test = joblib.load(cache_dir / f'y_test.joblib')

logger = Logger(models_dir / "train.log")

for c in CLASSIFIERS_TO_TRAIN:
    c.X_train = X_train if c.scaling_type == ScalingType.RAW else X_train_norm if c.scaling_type == ScalingType.NORM else X_train_scale
    c.X_test = X_test if c.scaling_type == ScalingType.RAW else X_test_norm if c.scaling_type == ScalingType.NORM else X_test_scale
    c.y_train = y_train
    c.y_test = y_test

# for c in CLASSIFIERS_TO_TRAIN:
#     print(f'{c.name}({type(c.X_train)}): {c.X_train}')

# sys.exit(0)

def print_search_results(model_name, search):
    logger.writeln(f'\n=== {model_name} Best Params ===')
    for key, val in search.best_params_.items():
        logger.writeln(f'{key}: {val}')

def test(c):
    logger.writeln(f'\n=== {c.name} Test ===')
    y_pred = c.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names = labels)
    logger.writeln(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels = labels)

    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)

    plt.tight_layout(pad=3.0)
    plt.savefig(models_dir / f'test_{c.name}.png', bbox_inches='tight')
    plt.close()

    return accuracy

for c in CLASSIFIERS_TO_TRAIN:
    start_time = time.time()
    c.model = c.func(c, CV, VERBOSE, print_search_results)
    elapsed_time = time.time() - start_time
    logger.writeln(f"{c.name} took {elapsed_time:.2f} seconds or {elapsed_time // 60} minutes.")
    
    c.accuracy = test(c)
    joblib.dump(c.model, models_dir / f'model_{c.name}.joblib')

best_c = max(CLASSIFIERS_TO_TRAIN, key=lambda c: c.accuracy if c.accuracy != None else 0.0)
# model_global model_general_pop model_rock model_edm
joblib.dump(best_c.model, cache_dir / 'model_global.joblib')
