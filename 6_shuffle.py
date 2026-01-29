import gc
import shutil
import numpy as np
import pandas as pd
import global_params as g

g.DATA_BATCH_SIZE = 14_000
BUCKET_COUNT = 1_000

g.TEMP_DIR.mkdir(exist_ok=True)

def save_bucket(data, bucket_id):
    bucket_dir = g.TEMP_DIR / f"bucket_{bucket_id}"
    bucket_dir.mkdir(exist_ok=True)
    part_id = len(list(bucket_dir.glob("*.parquet")))
    data.to_parquet(bucket_dir / f"part-{part_id:05d}.parquet", index=False)

def load_bucket(bucket_id):
    bucket_dir = g.TEMP_DIR / f"bucket_{bucket_id}"
    parts = list(bucket_dir.glob("*.parquet"))
    if not parts:
        return None
    data = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return data

for data_set_type in g.DataSetType:
    for data_path in g.iter_data_paths(5, data_set_type):
        batch = g.load_data(data_path)
        batch["_bucket"] = np.random.randint(0, BUCKET_COUNT, size=len(batch))

        for bucket_id, data in batch.groupby("_bucket"):
            save_bucket(data.drop(columns="_bucket"), bucket_id, data_set_type)

        del batch
        gc.collect()

    data = None
    batch_idx = 0
    for bucket_id in range(BUCKET_COUNT):
        bucket = load_bucket(bucket_id)
        if bucket is None:
            continue

        if data is None:
            data = pd.concat([data, bucket], ignore_index=True)
        else:
            data = bucket
        
        if len(data.index) > g.DATA_BATCH_SIZE:
            data = data.sample(frac=1).reset_index(drop=True)
            g.save_data(data, 6, data_set_type, batch_idx)
            batch_idx += 1

            data = None
            gc.collect()
        
        del bucket
        gc.collect()
    
    for path in g.TEMP_DIR.iterdir():
        shutil.rmtree(path)

g.TEMP_DIR.rmdir()
