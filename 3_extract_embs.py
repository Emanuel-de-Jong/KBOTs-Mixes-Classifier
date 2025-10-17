import pandas as pd
import numpy as np
import joblib
import Utils
import os
from pathlib import Path
from Mert import Mert
from tqdm import tqdm

MAX_CHUNKS_TRAIN = 12
MAX_CHUNKS_TEST = 5

cache_dir = Path("cache")
songs_train = pd.read_csv(cache_dir / "labels_train.csv")
songs_test = pd.read_csv(cache_dir / "labels_test.csv")

mert = Mert()

embs_train, embs_test = None, None
if os.path.exists(cache_dir / "embs_train_temp.joblib"):
    embs_train = joblib.load(cache_dir / "embs_train_temp.joblib")
    embs_test = joblib.load(cache_dir / "embs_train_temp.joblib")

def extract(songs, max_chunks):
    embs, labels = [], []
    for _, song in tqdm(songs.iterrows(), total=len(songs)):
        song_embs = mert.run(song.filepath, max_chunks)
        if song_embs is None:
            continue
        
        for emb in song_embs:
            if not isinstance(emb, np.ndarray):
                print(f"Skipping emb from {song.filepath}: returned {type(emb)} instead of ndarray.")
                continue
            if emb.shape != (25, Mert.TIME_STEPS, 1024):
                print(f"Skipping emb from {song.filepath}: wrong shape {emb.shape}.")
                continue

            embs.append(emb)
            labels.append(int(song.label))

    return np.stack(embs), pd.Series(labels)

if embs_train is None:
    embs_train, labels = extract(songs_train, MAX_CHUNKS_TRAIN)
    joblib.dump(labels, cache_dir / "labels_train.joblib")
    joblib.dump(embs_train, cache_dir / "embs_train_temp.joblib")

embs_train = Utils.preprocess(embs_train)
joblib.dump(embs_train, cache_dir / "embs_train.joblib")

if embs_test is None:
    embs_test, labels = extract(songs_test, MAX_CHUNKS_TEST)
    joblib.dump(labels, cache_dir / "labels_test.joblib")
    joblib.dump(embs_test, cache_dir / "embs_test_temp.joblib")

embs_test = Utils.preprocess(embs_test)
joblib.dump(embs_test, cache_dir / "embs_test.joblib")
