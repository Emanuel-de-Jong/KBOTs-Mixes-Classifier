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



def train_KNeighbors(c, CV, VERBOSE, print_search_results):
    model = KNeighborsClassifier(n_jobs=-1)

    # search_params = [
    #     {
    #         'metric': ['minkowski'],
    #         'n_neighbors': [3, 5, 7, 9],
    #         'weights': ['uniform', 'distance'],
    #         'p': [1.0, 2.0, 3.0],
    #     },
    #     {
    #         'metric': ['cosine'],
    #         'n_neighbors': [3, 5, 7, 9],
    #         'weights': ['uniform', 'distance'],
    #     },
    # ]

    search_params = [
        {
            'metric': ['minkowski'],
            'n_neighbors': [3],
            'weights': ['distance'],
            'p': [1.0],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_SVC(c, CV, VERBOSE, print_search_results):
    model = SVC(random_state=1)

    # search_params = [
    #     {
    #         'kernel': ['linear'],
    #         'C': [0.8, 1.0, 1.2],
    #         'break_ties': [True, False],
    #     },
    #     {
    #         'kernel': ['poly'],
    #         'C': [0.8, 1.0, 1.2],
    #         'degree': [2, 3, 4, 5, 6, 7],
    #         'gamma': ['scale', 'auto'],
    #         'coef0': [0.0, 0.2, 0.6, 1.0, 1.2],
    #         'break_ties': [True, False],
    #     },
    #     {
    #         'kernel': ['rbf'],
    #         'C': [0.8, 1.0, 1.2],
    #         'gamma': ['scale', 'auto'],
    #         'break_ties': [True, False],
    #     },
    #     {
    #         'kernel': ['sigmoid'],
    #         'C': [0.8, 1.0, 1.2],
    #         'gamma': ['scale', 'auto'],
    #         'coef0': [0.0, 0.2, 0.6, 1.0, 1.2],
    #         'break_ties': [True, False],
    #     },
    # ]

    search_params = [
        {
            'kernel': ['poly'],
            'C': [1.2],
            'degree': [5],
            'gamma': ['scale'],
            'coef0': [1.0],
            'break_ties': [False],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_GaussianNB(c, CV, VERBOSE, print_search_results):
    model = GaussianNB()

    # search_params = [
    #     {
    #         'var_smoothing': [
    #             1e-1,
    #             1e-3,
    #             1e-4,
    #             1e-5,
    #             1e-6,
    #             1e-7,
    #             1e-8, 4e-8, 8e-8, 9e-8,
    #             1e-9, 2e-9, 4e-9, 8e-9,
    #             1e-10, 5e-10],
    #     },
    # ]

    search_params = [
        {
            'var_smoothing': [1e-5],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_LogisticRegression(c, CV, VERBOSE, print_search_results):
    model = LogisticRegression(random_state=1, n_jobs=-1, max_iter=2000)

    # search_params = [
    #     {
    #         'penalty': ['l2'],
    #         'C': [0.8, 1.0, 1.2],
    #         'fit_intercept': [True, False],
    #         'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
    #         'class_weight': [None, 'balanced'],
    #     },
    #     {
    #         'penalty': ['l1'],
    #         'C': [0.8, 1.0, 1.2],
    #         'fit_intercept': [True, False],
    #         'solver': ['saga'],
    #         'class_weight': [None, 'balanced'],
    #     },
    #     {
    #         'penalty': ['elasticnet'],
    #         'C': [0.8, 1.0, 1.2],
    #         'fit_intercept': [True, False],
    #         'solver': ['saga'],
    #         'l1_ratio': [0.25, 0.5, 0.75]
    #         'class_weight': [None, 'balanced'],
    #     },
    # ]

    search_params = [
        {
            'penalty': ['l1'],
            'C': [1.0],
            'fit_intercept': [True],
            'solver': ['saga'],
            'class_weight': [None],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_MLP(c, CV, VERBOSE, print_search_results):
    model = MLPClassifier(random_state=1, early_stopping=True)

    # search_params = [
    #     {
    #         'solver': ['adam'],
    #         'max_iter': [500],
    #         'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
    #         'activation': ['relu', 'logistic', 'tanh'],
    #         'alpha': [5e-5, 1e-6, 5e-6],
    #         'epsilon': [8e-7, 1e-8, 2e-9],
    #         'learning_rate_init': [0.0005, 0.001, 0.005],
    #     },
    #     {
    #         'solver': ['lbfgs'],
    #         'max_iter': [1000],
    #         'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
    #         'activation': ['relu', 'logistic', 'tanh'],
    #         'alpha': [5e-5, 1e-6, 5e-6],
    #     },
    #     {
    #         'solver': ['sgd'],
    #         'max_iter': [500],
    #         'hidden_layer_sizes': [(100,), (50, 50), (25,50,25)],
    #         'activation': ['relu', 'logistic', 'tanh'],
    #         'alpha': [5e-5, 1e-6, 5e-6],
    #         'learning_rate': ['constant', 'adaptive'],
    #         'learning_rate_init': [0.0005, 0.001, 0.005],
    #         'momentum': [0.88, 0.9, 0.92],
    #     },
    # ]

    search_params = [
        {
            'solver': ['adam'],
            'max_iter': [500],
            'hidden_layer_sizes': [(100,)],
            'activation': ['tanh'],
            'alpha': [5e-5],
            'epsilon': [8e-7],
            'learning_rate_init': [0.005],
        },
    ]

    search = GridSearchCV(
        model,
        search_params,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_DecisionTree(c, CV, VERBOSE, print_search_results):
    model = DecisionTreeClassifier(random_state=1)

    # search_params = [
    #     {
    #         'criterion': ['gini', 'entropy'],
    #         'splitter': ['best', 'random'],
    #         'max_depth': [None, 5, 20, 200],
    #         'min_samples_split': [2, 5, 20],
    #         'min_samples_leaf': [1, 2, 3],
    #         'max_features': [None, 'sqrt', 'log2', 0.01, 0.1, 0.5],
    #         'class_weight': [None, 'balanced'],
    #     },
    # ]

    search_params = [
        {
            'criterion': ['entropy'],
            'splitter': ['best'],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': [None],
            'class_weight': [None],
        },
    ]

    search = GridSearchCV(model, search_params, cv=CV)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_RandomForest(c, CV, VERBOSE, print_search_results):
    model = RandomForestClassifier(random_state=1, n_jobs=-1)

    # search_params = [
    #     {
    #         'n_estimators': [50],
    #         'criterion': ['gini', 'entropy'],
    #         'max_depth': [5, 20, 200],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['sqrt', 0.01, 0.1, 0.5],
    #         'class_weight': [None, 'balanced'],
    #     },
    # ]

    search_params = [
        {
            'n_estimators': [50],
            'criterion': ['entropy'],
            'max_depth': [20],
            'min_samples_split': [2],
            'min_samples_leaf': [3],
            'max_features': [0.1],
            'class_weight': [None],
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

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_ExtraTrees(c, CV, VERBOSE, print_search_results):
    model = ExtraTreesClassifier(random_state=1, n_jobs=-1)

    # search_params = [
    #     {
    #         'n_estimators': [50],
    #         'criterion': ['gini', 'entropy'],
    #         'max_depth': [5, 20, 200],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['sqrt', 0.01, 0.1, 0.5],
    #         'class_weight': [None, 'balanced'],
    #     },
    # ]

    search_params = [
        {
            'n_estimators': [50],
            'criterion': ['entropy'],
            'max_depth': [20],
            'min_samples_split': [5],
            'min_samples_leaf': [3],
            'max_features': [0.5],
            'class_weight': [None],
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

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_

def train_GradientBoosting(c, CV, VERBOSE, print_search_results):
    model = GradientBoostingClassifier(random_state=1)

    # search_params = [
    #     {
    #         'loss': ['log_loss'],
    #         'learning_rate': [0.05, 0.1, 0.15],
    #         'subsample': [1.0, 0.8],
    #         'criterion': ['friedman_mse', 'squared_error'],
    #         'min_samples_split': [2, 4, 6],
    #         'max_depth': [2, 3, 4],
    #         'max_features': ['sqrt', 0.1, 0.5],
    #     },
    # ]

    search_params = [
        {
            'loss': ['log_loss'],
            'learning_rate': [0.1],
            'subsample': [1.0],
            'criterion': ['friedman_mse'],
            'min_samples_split': [2],
            'max_depth': [3],
            'max_features': ['sqrt'],
        },
    ]

    search = RandomizedSearchCV(
        model,
        search_params,
        n_iter=25,
        cv=CV,
        n_jobs=-1)
    
    if VERBOSE:
        search.verbose = 3

    search.fit(c.X_train, c.y_train)
    print_search_results(c.name, search)

    return search.best_estimator_
