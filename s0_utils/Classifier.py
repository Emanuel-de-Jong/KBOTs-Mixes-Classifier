import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import joblib
import s0_utils.global_params as g
from keras.models import load_model
from s0_utils.Mert import Mert

class Classifier():
    def __init__(self, name="global", mert=None):
        self.name = name
        if mert is None:
            mert = Mert()
        self.mert = mert
        self.model = load_model(g.MODELS_DIR / f"model_{name}.keras")
        self.labels = joblib.load(g.MODELS_DIR / f"labels_{name}.joblib")
        self.scale_tools = joblib.load(g.MODELS_DIR / f"scale_tools_{name}.joblib")
    
    def infer(self, path, embs=None):
        if embs is None:
            embs = self.mert.run(path)
            if embs is None or len(embs) == 0:
                return None, None
            
            embs = self.scale_embs(embs)
            embs = embs.transpose(0, 1, 3, 2)
        
        embs_probs = self.model.predict(embs)
        
        probs_avg = np.mean(embs_probs, axis=0)
        top_indices = probs_avg.argsort()[::-1][:5]

        results = []
        for idx in top_indices:
            prob_to_percent = int(probs_avg[idx] * 10000) / 100.0
            results.append((self.labels[idx], prob_to_percent))

        return results, embs
    
    def scale_embs(self, embs):
        clip_min = self.scale_tools["clip_min"]
        clip_max = self.scale_tools["clip_max"]

        range_vals = clip_max - clip_min
        range_vals[range_vals == 0] = 1.0

        embs = np.clip(
            embs,
            clip_min[None, :, None],
            clip_max[None, :, None]
        )

        embs = (embs - clip_min[None, :, None]) / range_vals[None, :, None]
        embs = embs * 2.0 - 1.0

        return embs.astype(np.float32, copy=False)
    
    def print_top(self, top):
        for i in range(len(top)):
            # if i >= 3:
            #     break

            label, val = top[i]
            print(f"{i+1}. {label}: {val:.2f}%")
