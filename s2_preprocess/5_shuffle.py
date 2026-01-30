import gc
import shutil
import joblib
import numpy as np
import pandas as pd
import s0_utils.global_params as g
from tqdm import tqdm

g.DATA_BATCH_SIZE = 14_000
BUCKET_COUNT = 500

def clear_temp_dir():
    for path in g.TEMP_DIR.iterdir():
        shutil.rmtree(path)

if g.TEMP_DIR.exists():
    clear_temp_dir()

g.TEMP_DIR.mkdir(exist_ok=True)

def save_bucket(data, bucket_id):
    bucket_dir = g.TEMP_DIR / f"bucket_{bucket_id}"
    bucket_dir.mkdir(exist_ok=True)
    part_id = len(list(bucket_dir.glob("*.joblib")))
    joblib.dump(data, bucket_dir / f"part-{part_id:05d}.joblib")

def load_bucket(bucket_id):
    bucket_dir = g.TEMP_DIR / f"bucket_{bucket_id}"
    parts = list(bucket_dir.glob("*.joblib"))
    if not parts:
        return None
    
    return pd.concat([joblib.load(p) for p in parts], ignore_index=True)

for data_set_type in tqdm(
        g.DataSetType,
        desc="Data Sets",
        position=0):
    step = 3 if data_set_type == g.DataSetType.test else 4
    data_paths = list(g.iter_data_paths(step, data_set_type))
    for data_path in tqdm(
            data_paths,
            desc="Batches",
            position=1,
            leave=False):
        batch = g.load_data(data_path)
        batch["_bucket"] = np.random.randint(0, BUCKET_COUNT, size=len(batch))

        for bucket_id, data in batch.groupby("_bucket"):
            save_bucket(data.drop(columns="_bucket"), bucket_id)

        del batch
        gc.collect()

    data = None
    batch_idx = 0
    for bucket_id in np.random.permutation(BUCKET_COUNT):
        bucket = load_bucket(bucket_id)
        if bucket is None:
            continue

        if data is None:
            data = bucket
        else:
            data = pd.concat([data, bucket], ignore_index=True)
        
        if len(data.index) >= g.DATA_BATCH_SIZE:
            data = data.sample(frac=1).reset_index(drop=True)
            g.save_zarr(data, 5, data_set_type, batch_idx)
            batch_idx += 1

            data = None
            gc.collect()
        
        del bucket
    
    if data is not None and not data.empty:
        data = data.sample(frac=1).reset_index(drop=True)
        g.save_zarr(data, 5, data_set_type, batch_idx)
    
    clear_temp_dir()

g.TEMP_DIR.rmdir()
