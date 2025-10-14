import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pathlib import Path

cache_dir = Path("cache")
X = np.load(cache_dir / "X_emb.npy")
y = pd.read_csv(cache_dir / "y_labels.csv").iloc[:, 0].astype(int)

with open(cache_dir / "num_to_label.json", "r") as f:
    labels = json.load(f)

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)

models = {}

train_distribution = y_train.value_counts()
print(train_distribution)

def print_grid_search_results(grid_search, params):
    print('Grid search best params')
    for key in params.keys():
        print(f'{key}: {grid_search.best_params_[key]}')

model = KNeighborsClassifier(n_jobs=-1)

grid_search_params = {
    'n_neighbors': [3, 5],
    'metric': ['cosine', 'minkowski'],
    'weights': ['uniform', 'distance'],
    }

grid_search = GridSearchCV(model, grid_search_params, cv=5)
grid_search.fit(X_train, y_train)
print_grid_search_results(grid_search, grid_search_params)

models['knn'] = grid_search.best_estimator_

def test(model_name):
    print(model_name)

    model = models[model_name]
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names = labels)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels = labels)
    disp.plot()

    plt.xticks(rotation=90)
    plt.show()

for model_name in models.keys():
    test(model_name)
