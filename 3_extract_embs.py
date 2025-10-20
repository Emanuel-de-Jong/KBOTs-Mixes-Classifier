import pandas as pd
import numpy as np
import global_params as g
from sklearn.utils import resample
from Mert import Mert
from tqdm import tqdm

MAX_CHUNKS_TRAIN = 18
MAX_CHUNKS_TEST = MAX_CHUNKS_TRAIN

songs_train = pd.read_csv(g.CACHE_DIR / "labels_train.csv")
songs_test = pd.read_csv(g.CACHE_DIR / "labels_test.csv")

mert = Mert()

def extract(data, data_set_type):
    songs = songs_train if data_set_type == g.DataSetType.train else songs_test
    max_chunks = MAX_CHUNKS_TRAIN if data_set_type == g.DataSetType.train else MAX_CHUNKS_TEST

    for _, song in tqdm(songs.iterrows(), total=len(songs)):
        song_label = int(song.label)
        song_name = g.get_song_name(song.song)
        song_embs = mert.run(song.song, max_chunks)
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
            if emb.shape != (Mert.TIME_STEPS, 1024, 25):
                print(f"Skipping emb from {song.filepath}: wrong shape {emb.shape}.")
                continue

            data.append({
                'data_set': data_set_type,
                'label': song_label,
                'song': song_name,
                'data': emb})

data = []

extract(data, g.DataSetType.train)
extract(data, g.DataSetType.test)

g.data = pd.DataFrame(data)
g.save_data(3)
