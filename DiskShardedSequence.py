import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import random
import zarr
import gc
import global_params as g
from keras.utils import to_categorical, Sequence

class DiskShardedSequence(Sequence):
    def __init__(self, shard_paths, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.shard_paths = shard_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = []
        for shard_id, path in enumerate(shard_paths):
            z = zarr.open(path, mode="r")
            n = z["label"].shape[0]
            for i in range(n):
                self.index.append((shard_id, i))

        self.current_shard_id = None
        self.current_zarr = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.index) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)

    def load_shard(self, shard_id):
        if self.current_shard_id != shard_id:
            self.current_zarr = zarr.open(self.shard_paths[shard_id], mode="r")
            self.current_shard_id = shard_id
            gc.collect()

    def __getitem__(self, idx):
        batch = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []
        for shard_id, row_idx in batch:
            self.load_shard(shard_id)
            X.append(self.current_zarr["data"][row_idx])
            y.append(self.current_zarr["label"][row_idx])

        X = np.stack(X)
        y = to_categorical(np.array(y), g.LABEL_COUNT)
        return X, y
