import pandas as pd
import numpy as np
import global_params as g
from Mert import Mert
from tqdm import tqdm

MAX_CHUNKS_TRAIN = 18
MAX_CHUNKS_TEST = -1

SONGS_BATCH_SIZE = 30

songs_train = pd.read_csv(g.CACHE_DIR / "labels_train.csv")
songs_test = pd.read_csv(g.CACHE_DIR / "labels_test.csv")

mert = Mert()

def extract(data_set_type):
    songs = songs_train if data_set_type == g.DataSetType.train else songs_test
    max_chunks = MAX_CHUNKS_TRAIN if data_set_type == g.DataSetType.train else MAX_CHUNKS_TEST

    batches = [songs.iloc[i:i+SONGS_BATCH_SIZE] for i in range(0, len(songs), SONGS_BATCH_SIZE)]
    for i in tqdm(
            range(len(batches)),
            desc="Batches",
            position=0):
        batch = batches[i]

        data = []
        for _, song in tqdm(
                batch.iterrows(),
                total=len(batch),
                desc="Songs",
                position=1,
                leave=False):
            song_label = int(song.label)
            song_name = g.get_song_name(song.song)
            song_embs = mert.run(song.song, max_chunks)
            if song_embs is None:
                continue
            
            for emb in song_embs:
                if not isinstance(emb, np.ndarray):
                    print(f"Skipping emb from {song.filepath}: returned {type(emb)} instead of ndarray.")
                    continue
                if emb.shape != (Mert.TIME_STEPS, 1024, 25):
                    print(f"Skipping emb from {song.filepath}: wrong shape {emb.shape}.")
                    continue

                data.append({
                    'label': song_label,
                    'song': song_name,
                    'is_public': song.is_public,
                    'data': emb})
        
        g.save_data(pd.DataFrame(data), 3, data_set_type, i)

extract(g.DataSetType.train)
extract(g.DataSetType.test)
