import gc
import numpy as np
import pandas as pd
import global_params as g

g.DATA_BATCH_SIZE = 14_000

BUCKET_COUNT = 5_000

g.TEMP_DIR.mkdir(exist_ok=True)

def save_bucket(df, bucket_id):
    bucket_path = g.TEMP_DIR / f"{bucket_id}.parquet"

    if bucket_path.exists():
        existing = pd.read_parquet(bucket_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(bucket_path, index=False)
    else:
        df.to_parquet(bucket_path, index=False)

for data_set_type in g.DataSetType:
    for data_path in g.iter_data_paths(5, data_set_type):
        batch = g.load_data(data_path)

        buckets = np.random.randint(0, BUCKET_COUNT, size=len(batch))
        batch["_buckets"] = buckets

        for bucket_id in range(BUCKET_COUNT):
            bucket = batch[batch["_buckets"] == bucket_id].drop(columns="_buckets")
            if not bucket.empty:
                save_bucket(bucket, bucket_id)

        del batch
    
    for bucket_id in range(BUCKET_COUNT):
        bucket_path = g.TEMP_DIR / f"{bucket_id}.parquet"
        bucket = pd.read_parquet(bucket_path)
        bucket = bucket.sample(frac=1).reset_index(drop=True)
        g.save_data_batched(bucket, 6, data_set_type)
        del bucket
    
    for bucket_path in g.TEMP_DIR.iterdir():
        bucket_path.unlink()

g.TEMP_DIR.rmdir()
