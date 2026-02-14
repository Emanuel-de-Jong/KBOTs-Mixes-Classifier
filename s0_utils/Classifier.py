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
            embs = self.reshape_data(embs)
        
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
        scaler = self.scale_tools["scaler"]

        embs_2d = embs.transpose(0, 1, 3, 2).reshape(-1, embs.shape[2])
        embs_2d = np.clip(
            embs_2d,
            clip_min,
            clip_max
        )

        embs_scaled = scaler.transform(embs_2d)
        embs_scaled = embs_scaled.reshape(
            embs.shape[0],
            embs.shape[1],
            embs.shape[3],
            embs.shape[2]
        ).transpose(0, 1, 3, 2)

        return embs_scaled.astype(np.float32, copy=False)
    
    def reshape_data(self, embs):
        axis_mapping = {
            g.DataSectionType.time: 1,
            g.DataSectionType.layer: 2,
            g.DataSectionType.feature: 3
        }

        transpose_order = [0]
        for section in g.DATA_ORDER:
            transpose_order.append(axis_mapping[section])

        return np.transpose(embs, transpose_order)
    
    def print_top(self, top):
        for i in range(len(top)):
            # if i >= 3:
            #     break

            label, val = top[i]
            print(f"{i+1}. {label}: {val:.2f}%")
