import pandas as pd
import numpy as np
from pathlib import Path
from Mert import Mert
from tqdm import tqdm

MAX_CHUNKS_TRAIN = 12
MAX_CHUNKS_TEST = 5

cache_dir = Path("cache")
songs_train = pd.read_csv(cache_dir / "labels_train.csv")
songs_test = pd.read_csv(cache_dir / "labels_test.csv")

mert = Mert()

def extract(songs, max_chunks):
    embeddings, labels = [], []
    for _, row in tqdm(songs.iterrows(), total=len(songs)):
        chunk_data = mert.run(row.filepath, max_chunks)
        if chunk_data is None:
            continue
        
        for vec in chunk_data:
            if not isinstance(vec, np.ndarray):
                print(f"Skipping chunk from {row.filepath}: returned {type(vec)} instead of ndarray.")
                continue
            if vec.shape != (1024,):
                print(f"Skipping chunk from {row.filepath}: wrong shape {vec.shape}.")
                continue

            embeddings.append(vec)
            labels.append(row.label)

    return np.stack(embeddings), pd.Series(labels)

X, labels = extract(songs_train, MAX_CHUNKS_TRAIN)
np.save(cache_dir / "X_emb_train.npy", X)
labels.to_csv(cache_dir / "y_labels_train.csv", index=False, header=["labels"])

X, labels = extract(songs_test, MAX_CHUNKS_TEST)
np.save(cache_dir / "X_emb_test.npy", X)
labels.to_csv(cache_dir / "y_labels_test.csv", index=False, header=["labels"])
