import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import global_params as g
from keras.models import load_model
from Mert import Mert

class Classifier():
    def __init__(self, model_name="global", mert=None):
        self.model_name = model_name
        if mert is None:
            mert = Mert()
        self.mert = mert
        self.model = load_model(g.CACHE_DIR / "model_global.keras")
    
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
            results.append((g.labels[idx], prob_to_percent))

        return results, embs
    
    def print_top(self, top):
        for i in range(len(top)):
            # if i >= 3:
            #     break

            label, val = top[i]
            print(f"{i+1}. {label}: {val:.2f}%")
