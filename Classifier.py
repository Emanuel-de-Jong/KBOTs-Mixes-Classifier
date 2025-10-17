import numpy as np
import pandas as pd
import joblib
import Utils
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import load_model
from sklearn.svm import SVC
from pathlib import Path
from Mert import Mert

class Classifier():
    def __init__(self, model_name="global", mert=None):
        self.model_name = model_name
        if mert is None:
            mert = Mert()
        self.mert = mert
        self.cache_dir = Path("cache")
        self.labels = np.unique(pd.read_json(self.cache_dir / "num_to_label.json"))
        self.model = load_model("model/model.keras")
    
    def infer(self, path, embs=None):
        if embs is None:
            embs = self.mert.run(path)
            if embs is None or len(embs) == 0:
                return None
            
            embs = Utils.preprocess(embs)
            
        all_probs = self.model.predict(embs)
        return None

        all_probs = np.zeros(len(self.labels))
        for emb in embs:
            emb = emb.reshape(1, -1)
            probs = self.model.predict_proba(emb)[0]

            all_probs += probs

        all_probs /= len(embs)

        top_indices = all_probs.argsort()[::-1][:5]

        results = []
        for idx in top_indices:
            prob_to_percent = int(all_probs[idx] * 10000) / 100.0
            results.append((self.labels[idx], prob_to_percent))

        return results, embs
    
    def print_top(self, top):
        for i in range(len(top)):
            # if i >= 3:
            #     break

            label, val = top[i]
            print(f"{i+1}. {label}: {val:.2f}%")
