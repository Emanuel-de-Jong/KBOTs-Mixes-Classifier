import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import random
import global_params as g
from keras.utils import to_categorical, Sequence
import joblib
import gc

class DiskShardedSequence(Sequence):
    def __init__(self, shard_paths, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.shard_paths = shard_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = []
        for shard_id, p in enumerate(shard_paths):
            df = joblib.load(p)
            for i in range(len(df)):
                self.index.append((shard_id, i))
            del df

        self.current_shard_id = None
        self.current_df = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.index) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)

    def __getitem__(self, idx):
        batch = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []
        for shard_id, row_idx in batch:
            if shard_id != self.current_shard_id:
                if self.current_df is not None:
                    del self.current_df
                    gc.collect()
                self.current_df = joblib.load(self.shard_paths[shard_id])
                self.current_shard_id = shard_id

            row = self.current_df.iloc[row_idx]
            X.append(row["data"])
            y.append(row["label"])

        X = np.stack(X)
        y = to_categorical(np.array(y), g.LABEL_COUNT)
        return X, y
