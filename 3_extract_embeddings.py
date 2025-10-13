import pandas as pd
import numpy as np
from pathlib import Path
from Mert import Mert
from tqdm import tqdm

mert = Mert()

cache_dir = Path("cache")
df = pd.read_csv(cache_dir / "labels.csv")
embs, labels = [], []

song_batch_count = 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    song_batch_count += 1
    # if song_batch_count > 25:
    #     break

    vec = mert.run(row.filepath)

    embs.append(vec)
    labels.append(row.label)

X = np.stack(embs)

pd.Series(labels).to_csv(cache_dir / "y_labels.csv", index=False, header=["labels"])
np.save(cache_dir / "X_emb.npy", X)

print("Saved X_emb.npy and y_labels.csv")
