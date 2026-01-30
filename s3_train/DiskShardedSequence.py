import os

os.environ["KERAS_BACKEND"] = "torch"

import random
import zarr
import s0_utils.global_params as g
from keras.utils import to_categorical, Sequence

class DiskShardedSequence(Sequence):
    def __init__(self, shard_paths, batch_size=256, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.shard_paths = shard_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.shards = []
        for shard_id, path in enumerate(shard_paths):
            z = zarr.open(path, mode="r")
            n = z["label"].shape[0]
            self.shards.append((shard_id, n))

        self.order = {}
        for shard_id, n in self.shards:
            self.order[shard_id] = []
            for start in range(0, n, self.batch_size):
                self.order[shard_id].append(start)

        self.shard_order = list(self.order.keys())
        self.current_shard_id = None
        self.current_zarr = None
        self.flat_order = []
        self.on_epoch_end()

    def __len__(self):
        return len(self.flat_order)

    def on_epoch_end(self):
        self.flat_order = []

        if self.shuffle:
            random.shuffle(self.shard_order)

        for shard_id in self.shard_order:
            starts = self.order[shard_id]
            if self.shuffle:
                random.shuffle(starts)

            for start in starts:
                self.flat_order.append((shard_id, start))

    def load_shard(self, shard_id):
        if self.current_shard_id != shard_id:
            self.current_zarr = zarr.open(self.shard_paths[shard_id], mode="r")
            self.current_shard_id = shard_id

    def __getitem__(self, idx):
        shard_id, start = self.flat_order[idx]
        self.load_shard(shard_id)

        z = self.current_zarr
        end = min(start + self.batch_size, z["label"].shape[0])

        X = z["data"][start:end]
        y = z["label"][start:end]

        y = to_categorical(y, g.LABEL_COUNT)
        return X, y
