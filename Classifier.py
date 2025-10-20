import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
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
        self.model = load_model(self.cache_dir / "model_global.keras")
    
    def infer(self, path, embs=None):
        if embs is None:
            embs = self.mert.run(path)
            if embs is None or len(embs) == 0:
                return None
        
        embs_probs = self.model.predict(embs)
        
        probs_avg = np.mean(embs_probs, axis=0)
        top_indices = probs_avg.argsort()[::-1][:5]

        results = []
        for idx in top_indices:
            prob_to_percent = int(probs_avg[idx] * 10000) / 100.0
            results.append((self.labels[idx], prob_to_percent))

        return results, embs
    
    def print_top(self, top):
        for i in range(len(top)):
            # if i >= 3:
            #     break

            label, val = top[i]
            print(f"{i+1}. {label}: {val:.2f}%")
