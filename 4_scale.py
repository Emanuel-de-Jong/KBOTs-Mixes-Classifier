import pandas as pd
import numpy as np
import joblib
import gc
import os
import global_params as g
from sklearn.preprocessing import MinMaxScaler

SCALE_TOOLS_PATH = g.CACHE_DIR / f"scale_tools_{g.NAME}.joblib"
SCALE_BATCH_SIZE = 1000

g.load_data(3)

scale_tools = {}
is_scale_tools_loaded = os.path.exists(SCALE_TOOLS_PATH)
if is_scale_tools_loaded:
    scale_tools = joblib.load(SCALE_TOOLS_PATH)
else:
    all_values = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in g.data["data"]], axis=0)
    scale_tools = {
        "scaler": MinMaxScaler(feature_range=(-1, 1)),
        "clip_min": np.percentile(all_values, 1, axis=0),
        "clip_max": np.percentile(all_values, 99, axis=0),
    }

    del all_values
    gc.collect()

    print("Clipping ranges per feature:")
    print(pd.DataFrame({"clip_min": scale_tools["clip_min"], "clip_max": scale_tools["clip_max"]}))

data_count = len(g.data)

if not is_scale_tools_loaded:
    for start in range(0, data_count, SCALE_BATCH_SIZE):
        end = min(start + SCALE_BATCH_SIZE, data_count)
        batch = [g.data.at[i, "data"] for i in range(start, end)]

        batch_2d = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in batch], axis=0)
        batch_2d = np.clip(batch_2d, scale_tools["clip_min"], scale_tools["clip_max"])
        scale_tools["scaler"].partial_fit(batch_2d)

        del batch_2d, batch
        gc.collect()

for start in range(0, data_count, SCALE_BATCH_SIZE):
    end = min(start + SCALE_BATCH_SIZE, data_count)
    batch = [g.data.at[i, "data"] for i in range(start, end)]

    batch_2d = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in batch], axis=0)
    batch_2d = np.clip(batch_2d, scale_tools["clip_min"], scale_tools["clip_max"])
    batch_scaled_2d = scale_tools["scaler"].transform(batch_2d)

    offset = 0
    for i, arr in enumerate(batch):
        sz = np.prod(arr.shape[:-1])
        arr_scaled = batch_scaled_2d[offset:offset+sz].reshape(arr.shape)
        g.data.at[start + i, "data"] = arr_scaled
        offset += sz

    del batch, batch_2d, batch_scaled_2d
    gc.collect()

if not is_scale_tools_loaded:
    joblib.dump(scale_tools, SCALE_TOOLS_PATH)

g.save_data(4)
