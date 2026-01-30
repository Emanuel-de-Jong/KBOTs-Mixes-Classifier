import pandas as pd
import numpy as np
import joblib
import gc
import os
import s0_utils.global_params as g
from sklearn.preprocessing import MinMaxScaler

g.DATA_BATCH_SIZE = 7_000

SCALE_TOOLS_PATH = g.MODELS_DIR / f"scale_tools_{g.NAME}.joblib"

scale_tools = {}
is_scale_tools_loaded = os.path.exists(SCALE_TOOLS_PATH)

if is_scale_tools_loaded:
    scale_tools = joblib.load(SCALE_TOOLS_PATH)
else:
    sample_loaded = False
    for data_path in g.iter_data_paths(2, g.DataSetType.train):
        data = g.load_data(data_path)
        feature_dim = data.iloc[0]["data"].shape[-1]
        break

    clip_min = np.empty(feature_dim, dtype=np.float32)
    clip_max = np.empty(feature_dim, dtype=np.float32)

    for f in range(feature_dim):
        values = []

        for data_path in g.iter_data_paths(2, g.DataSetType.train):
            data = g.load_data(data_path)

            layer_vals = np.concatenate(
                [arr[..., f].reshape(-1) for arr in data["data"]],
                axis=0
            )
            values.append(layer_vals)

        values = np.concatenate(values, axis=0)
        clip_min[f] = np.percentile(values, 1)
        clip_max[f] = np.percentile(values, 99)

        del values
        gc.collect()

    scale_tools = {
        "scaler": MinMaxScaler(feature_range=(-1, 1)),
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    print("Clipping ranges per feature:")
    print(pd.DataFrame({
        "clip_min": clip_min,
        "clip_max": clip_max
    }))

    for data_path in g.iter_data_paths(2, g.DataSetType.train):
        data = g.load_data(data_path)

        all_values = np.concatenate(
            [arr.reshape(-1, arr.shape[-1]) for arr in data["data"]],
            axis=0
        )

        all_values = np.clip(
            all_values,
            clip_min,
            clip_max
        )

        scale_tools["scaler"].partial_fit(all_values)

        del all_values
        gc.collect()

    joblib.dump(scale_tools, SCALE_TOOLS_PATH)

for data_set_type in g.DataSetType:
    out_idx = 0
    out_rows = []

    for data_path in g.iter_data_paths(2, data_set_type):
        data = g.load_data(data_path)

        all_values = np.concatenate(
            [arr.reshape(-1, arr.shape[-1]) for arr in data["data"]],
            axis=0
        )

        all_values = np.clip(
            all_values,
            scale_tools["clip_min"],
            scale_tools["clip_max"]
        )

        all_scaled = scale_tools["scaler"].transform(all_values)

        offset = 0
        for _, row in data.iterrows():
            arr = row["data"]
            sz = np.prod(arr.shape[:-1])
            row["data"] = all_scaled[offset:offset + sz].reshape(arr.shape)
            offset += sz

            out_rows.append(row)

            if len(out_rows) >= g.DATA_BATCH_SIZE:
                g.save_data(pd.DataFrame(out_rows), 3, data_set_type, out_idx)
                out_idx += 1
                out_rows = []

        del all_values, all_scaled
        gc.collect()

    if out_rows:
        g.save_data(pd.DataFrame(out_rows), 3, data_set_type, out_idx)
