import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample
from pathlib import Path
from Mert import Mert
from tqdm import tqdm

MAX_CHUNKS_TRAIN = 18
MAX_CHUNKS_TEST = MAX_CHUNKS_TRAIN

cache_dir = Path("cache")
songs_train = pd.read_csv(cache_dir / "labels_train.csv")
songs_test = pd.read_csv(cache_dir / "labels_test.csv")

mert = Mert()

def extract(songs, max_chunks):
    embs, labels = [], []
    for _, song in tqdm(songs.iterrows(), total=len(songs)):
        song_embs = mert.run(song.filepath, max_chunks)
        if song_embs is None:
            continue

        # Fill with dupes so each song has the same amount of training data.
        # If used, remove dupes first when undersampling and make sure no dupes are in the validation split!
        # if len(song_embs) < max_chunks:
        #     fill_embs = resample(song_embs, replace=False, n_samples=max_chunks - len(song_embs), random_state=1)
        #     song_embs = np.concatenate((song_embs, fill_embs))
        
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

embs_train, labels = extract(songs_train, MAX_CHUNKS_TRAIN)
joblib.dump(labels, cache_dir / "labels_train.joblib")
joblib.dump(embs_train, cache_dir / "embs_train.joblib")

embs_test, labels = extract(songs_test, MAX_CHUNKS_TEST)
joblib.dump(labels, cache_dir / "labels_test.joblib")
joblib.dump(embs_test, cache_dir / "embs_test.joblib")
