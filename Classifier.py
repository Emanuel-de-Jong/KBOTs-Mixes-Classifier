import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pathlib import Path
from Mert import Mert

class Classifier():
    def __init__(self):
        self.mert = Mert()
        self.cache_dir = Path("cache")
        self.labels = np.unique(pd.read_json(self.cache_dir / "num_to_label.json"))
        self.model = joblib.load(self.cache_dir / "model.joblib")
    
    def infer(self, path):
        chunk_data = self.mert.run(path)
        if chunk_data is None or len(chunk_data) == 0:
            return None

        all_weighted_probs = np.zeros(len(self.labels))
        for vec in chunk_data:
            vec = vec.reshape(1, -1)
            probs = self.model.predict_proba(vec)[0]
            
            top3_indices = probs.argsort()[::-1][:3]
            weights = [0.5, 0.3, 0.2]
            
            for idx, weight in zip(top3_indices, weights):
                all_weighted_probs[idx] += probs[idx] * weight

        all_weighted_probs /= len(chunk_data)

        top_indices = all_weighted_probs.argsort()[::-1][:5]
        results = []
        for idx in top_indices:
            results.append((self.labels[idx], all_weighted_probs[idx]))

        return results
    
    def print_top(self, top):
        for i in range(len(top)):
            if i >= 3:
                break

            label, val = top[i]
            print(f"{i+1}. {label}: {val}")



def train_KNeighbors(X_train, y_train, models, CV, VERBOSE, print_search_results):
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
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_SVC(X_train, y_train, models, CV, VERBOSE, print_search_results):
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
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_GaussianNB(X_train, y_train, models, CV, VERBOSE, print_search_results):
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
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_LogisticRegression(X_train, y_train, models, CV, VERBOSE, print_search_results):
    model_name = 'LogisticRegression'
    model = LogisticRegression(random_state=1, n_jobs=-1, max_iter=2000)

    search_params = [
        {
            'penalty': ['l2'],
            'C': [0.8, 1.0, 1.2],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
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
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_MLP(X_train, y_train, models, CV, VERBOSE, print_search_results):
    model_name = 'MLP'
    model = MLPClassifier(random_state=1, early_stopping=True)

    search_params = [
        {
            'solver': ['adam'],
            'max_iter': [500],
            'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
            'activation': ['relu', 'logistic', 'tanh'],
            'alpha': [0.00005, 0.0001, 0.0005],
            'epsilon': [8e-7, 1e-8, 2e-9],
            'learning_rate_init': [0.0005, 0.001, 0.005],
        },
        {
            'solver': ['lbfgs'],
            'max_iter': [1000],
            'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
            'activation': ['relu', 'logistic', 'tanh'],
            'alpha': [0.00005, 0.0001, 0.0005],
        },
        {
            'solver': ['sgd'],
            'max_iter': [500],
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
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_DecisionTree(X_train, y_train, models, CV, VERBOSE, print_search_results):
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

    search = GridSearchCV(model, search_params, cv=CV)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_RandomForest(X_train, y_train, models, CV, VERBOSE, print_search_results):
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
        cv=CV,
        random_state=1,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(X_train, y_train)
    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name

def train_ExtraTrees(X_train, y_train, models, CV, VERBOSE, print_search_results):
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
        cv=CV,
        random_state=1,
        n_jobs=-1)
    search.fit(X_train, y_train)
    
    if VERBOSE:
        search.verbose = 3

    print_search_results(model_name, search)

    models[model_name] = search.best_estimator_
    return model_name
