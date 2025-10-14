import numpy as np
import pandas as pd
import joblib
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
