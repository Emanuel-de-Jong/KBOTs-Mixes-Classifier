import pandas as pd
import numpy as np
from pathlib import Path
from Mert import Mert
from tqdm import tqdm

mert = Mert()

cache_dir = Path("cache")
df = pd.read_csv(cache_dir / "labels.csv")
embeddings, labels = [], []

song_batch_count = 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    song_batch_count += 1
    # if song_batch_count > 25:
    #     break

    chunk_data = mert.run(row.filepath)
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

X = np.stack(embeddings)

pd.Series(labels).to_csv(cache_dir / "y_labels.csv", index=False, header=["labels"])
np.save(cache_dir / "X_emb.npy", X)

print("Saved X_emb.npy and y_labels.csv")
