import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Classifier
import joblib
import time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn, set_config
from sklearn.utils import resample
from pathlib import Path

# 0: As is
# 1: Undersampling
# 2: Oversampling
SAMPLING = 1
VERBOSE = False
CV = 4
TRAIN_FUNCTIONS = [
    Classifier.train_KNeighbors,
    Classifier.train_SVC,
    Classifier.train_GaussianNB,
    Classifier.train_LogisticRegression,
    Classifier.train_MLP,
    Classifier.train_DecisionTree,
    Classifier.train_RandomForest,
    Classifier.train_ExtraTrees,
    Classifier.train_GradientBoosting,
]

# patch_sklearn()
# set_config(target_offload="gpu")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
X = np.load(cache_dir / "X_emb.npy")
y = pd.read_csv(cache_dir / "y_labels.csv")["labels"].astype(int)
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))

def write(msg):
    with open(models_dir / f"train_{SAMPLING}.log", "a") as f:
        f.write(f"{msg}\n")
    print(msg)

test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)

if SAMPLING == 1:
    df = pd.DataFrame({'label': y_train.values})
    df['idx'] = df.index
    min_n = df['label'].value_counts().min()

    selected_idxs = (
        df.groupby('label')['idx']
        .apply(lambda idxs: resample(idxs, replace=False, n_samples=min_n, random_state=1))
        .explode()
        .astype(int)
        .values
    )

    X_train = X_train[selected_idxs]
    y_train = y_train.iloc[selected_idxs].reset_index(drop=True)
elif SAMPLING == 2:
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# train_distribution = y_train.value_counts()
# for label_num, count in train_distribution.items():
#     print(f"{labels[label_num]}: {count}")

def print_search_results(model_name, search):
    write(f'\n=== {model_name} Best Params ===')
    for key, val in search.best_params_.items():
        write(f'{key}: {val}')

def test(model_name):
    write(f'\n=== {model_name} Test ===')
    model = models[model_name]
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names = labels)
    write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels = labels)

    _, ax = plt.subplots(figsize=(20, 22), dpi=200)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)

    plt.tight_layout(pad=3.0)
    plt.savefig(models_dir / f'test_{model_name}_{SAMPLING}.png', bbox_inches='tight')
    plt.close()

models = {}
for train_func in TRAIN_FUNCTIONS:
    start = time.time()
    model_name = train_func(X_train, y_train, models, CV, VERBOSE, print_search_results)
    elapsed = time.time() - start
    write(f"{train_func.__name__} took {elapsed:.2f} seconds or {elapsed // 60} minutes.")
    
    test(model_name)
    joblib.dump(models[model_name], models_dir / f'model_{model_name}_{SAMPLING}.joblib')
