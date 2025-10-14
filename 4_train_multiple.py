import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
from sklearn.svm import SVC
from pathlib import Path

train_dir = Path("train")
train_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
unbalanced_X = np.load(cache_dir / "X_emb.npy")
unbalanced_y = pd.read_csv(cache_dir / "y_labels.csv")["labels"].astype(int)
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))

cv = 4
verbose = False

smallest_label_data_count = unbalanced_y.value_counts().min()

label_indices = defaultdict(list)
for idx, label in enumerate(unbalanced_y):
    label_indices[label].append(idx)

selected_indices = []
rng = np.random.default_rng(1)
for label, indices in label_indices.items():
    indices = np.array(indices)
    if len(indices) > smallest_label_data_count:
        chosen = rng.choice(indices, smallest_label_data_count, replace=False)
    else:
        chosen = indices
    selected_indices.extend(chosen)

selected_indices = np.array(selected_indices)
rng.shuffle(selected_indices)

X = unbalanced_X[selected_indices]
y = unbalanced_y.iloc[selected_indices].reset_index(drop=True)

# X = unbalanced_X
# y = unbalanced_y

test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)

def write(msg):
    with open(train_dir / "train.log", "a") as f:
        f.write(f"{msg}\n")
    print(msg)

models = {}

train_distribution = y_train.value_counts()
for label_num, count in train_distribution.items():
    print(f"{labels[label_num]}: {count}")

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
    disp.plot()

    plt.xticks(rotation=90)
    plt.savefig(train_dir / f'test_{model_name}.png')

def train_KNeighbors():
    model_name = 'KNeighbors'
    model = KNeighborsClassifier(n_jobs=-1)

    search_params = [
        {
            'metric': ['minkowski'],
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1.0, 2.0, 3.0],
        },
        {
            'metric': ['cosine'],
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=cv,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_SVC():
    model_name = 'SVC'
    model = SVC(random_state=1)

    search_params = [
        {
            'kernel': ['linear'],
            'C': [0.8, 1.0, 1.2],
            'break_ties': [True, False],
        },
        {
            'kernel': ['poly'],
            'C': [0.8, 1.0, 1.2],
            'degree': [2, 3, 4, 5, 6, 7],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.2, 0.6, 1.0, 1.2],
            'break_ties': [True, False],
        },
        {
            'kernel': ['rbf'],
            'C': [0.8, 1.0, 1.2],
            'gamma': ['scale', 'auto'],
            'break_ties': [True, False],
        },
        {
            'kernel': ['sigmoid'],
            'C': [0.8, 1.0, 1.2],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.2, 0.6, 1.0, 1.2],
            'break_ties': [True, False],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=cv,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_GaussianNB():
    model_name = 'GaussianNB'
    model = GaussianNB()

    search_params = [
        {
            'var_smoothing': [
                1e-1,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8, 4e-8, 8e-8, 9e-8,
                1e-9, 2e-9, 4e-9, 8e-9,
                1e-10, 5e-10],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=cv,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_LogisticRegression():
    model_name = 'LogisticRegression'
    model = LogisticRegression(random_state=1, n_jobs=-1, max_iter=2000)

    search_params = [
        {
            'penalty': ['l2'],
            'C': [0.8, 1.0, 1.2],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        },
        {
            'penalty': ['l1'],
            'C': [0.8, 1.0, 1.2],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced'],
            'solver': ['saga'],
        },
        {
            'penalty': ['elasticnet'],
            'C': [0.8, 1.0, 1.2],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced'],
            'solver': ['saga'],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=cv,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_MLP():
    model_name = 'MLP'
    model = MLPClassifier(random_state=1, early_stopping=True, max_iter=500)

    search_params = [
        {
            'solver': ['adam'],
            'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
            'activation': ['relu', 'logistic', 'tanh'],
            'alpha': [0.00005, 0.0001, 0.0005],
            'epsilon': [8e-7, 1e-8, 2e-9],
            'learning_rate_init': [0.0005, 0.001, 0.005],
        },
        {
            'solver': ['lbfgs'],
            'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
            'activation': ['relu', 'logistic', 'tanh'],
            'alpha': [0.00005, 0.0001, 0.0005],
        },
        {
            'solver': ['sgd'],
            'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
            'activation': ['relu', 'logistic', 'tanh'],
            'alpha': [0.00005, 0.0001, 0.0005],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.0005, 0.001, 0.005],
            'momentum': [0.88, 0.9, 0.92],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=cv,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_DecisionTree():
    model_name = 'DecisionTree'
    model = DecisionTreeClassifier(random_state=1)

    search_params = [
        {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 20, 200],
            'min_samples_split': [2, 5, 20],
            'min_samples_leaf': [1, 2, 3],
            'max_features': [None, 'sqrt', 'log2', 0.01, 0.1, 0.5],
            'class_weight': [None, 'balanced'],
        },
    ]

    search = GridSearchCV(model, search_params, cv=cv)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_RandomForest():
    model_name = 'RandomForest'
    model = RandomForestClassifier(random_state=1, n_jobs=-1)

    search_params = [
        {
            'n_estimators': [50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 20, 200],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 0.01, 0.1, 0.5],
            'class_weight': [None, 'balanced'],
        },
    ]

    search = RandomizedSearchCV(model,
        search_params,
        n_iter=50,
        cv=cv,
        random_state=1,
        n_jobs=-1)
    
    if verbose:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_ExtraTrees():
    model_name = 'ExtraTrees'
    model = ExtraTreesClassifier(random_state=1, n_jobs=-1)

    search_params = [
            {
                'n_estimators': [50],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 20, 200],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['sqrt', 0.01, 0.1, 0.5],
                'class_weight': [None, 'balanced'],
            },
        ]

    search = RandomizedSearchCV(model,
        search_params,
        n_iter=50,
        cv=cv,
        random_state=1,
        n_jobs=-1)
    search.fit(X_train, y_train)
    
    if verbose:
        search.verbose = 3

    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

def train_GradientBoosting():
    model_name = 'GradientBoosting'
    model = GradientBoostingClassifier(random_state=1)

    search_params = [
        {
            'loss': ['log_loss'],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [1.0, 0.8],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split': [2, 4, 6],
            'max_depth': [2, 3, 4],
            'max_features': ['sqrt', 0.1, 0.5],
        },
    ]

    search = RandomizedSearchCV(
        model,
        search_params,
        n_iter=25,
        cv=cv,
        n_jobs=-1)
    search.fit(X_train, y_train)
    
    if verbose:
        search.verbose = 3

    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_

# train_KNeighbors()
# train_SVC()
# train_GaussianNB()
# train_LogisticRegression()
# train_MLP()
# train_DecisionTree()
# train_RandomForest()
# train_ExtraTrees()
# train_GradientBoosting()

for model_name in models.keys():
    test(model_name)

for name, model in models.items():
    joblib.dump(model, train_dir / f'model_{name}.joblib')
