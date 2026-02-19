import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import essentia
essentia.log.warningActive = False
import json
import numpy as np
from pathlib import Path
from essentia import Pool
from essentia.standard import MonoLoader, TensorflowPredictMAEST, TensorflowPredict
import s0_utils.global_params as g

class EssDiscogs():
    NAME = "ess_discogs_519"
    EMBEDDING_MODEL_PATH = g.MODELS_DIR / Path("discogs-maest-30s-pw-519l-2.pb")
    CLASSIFY_MODEL_PATH = g.MODELS_DIR / Path("genre_discogs519-discogs-maest-30s-pw-519l-1.pb")
    LABELS_PATH = g.MODELS_DIR / Path("genre_discogs519-discogs-maest-30s-pw-519l-1.json")

    def __init__(self):
        self.embedding_model = TensorflowPredictMAEST(graphFilename=str(self.EMBEDDING_MODEL_PATH), output="PartitionedCall/Identity_12")
        self.classify_model = TensorflowPredict(graphFilename=str(self.CLASSIFY_MODEL_PATH), inputs=["embeddings"], outputs=["PartitionedCall/Identity_1"])

        with open(self.LABELS_PATH, "r") as f:
            metadata = json.load(f)

        self.labels = metadata["classes"]
        self.pool = Pool()
    
    def get_embs(self, path):
        audio = MonoLoader(filename=str(path), sampleRate=16000, resampleQuality=4)()
        return self.embedding_model(audio)

    def infer(self, path, embs=None):
        if embs is None:
            embs = self.get_embs(path)

        self.pool.clear()
        self.pool.set("embeddings", embs)

        probs_avg = self.classify_model(self.pool)["PartitionedCall/Identity_1"]
        probs_avg = np.mean(probs_avg, axis=0).flatten()

        top_indices = np.argsort(probs_avg)[::-1][:5]

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
