import pandas as pd
import numpy as np
import joblib
import gc
import os
import s0_utils.global_params as g
from sklearn.preprocessing import MinMaxScaler

g.DATA_BATCH_SIZE = 7_000

# (min, max)
CLIPS_BY_LAYER = []
CLIPS_BY_LAYER[0] = (-7, 17)
CLIPS_BY_LAYER[1] = (-10, 15)
CLIPS_BY_LAYER[2] = (-10, 15)
CLIPS_BY_LAYER[3] = (-10, 15)
CLIPS_BY_LAYER[4] = (-10, 15)
CLIPS_BY_LAYER[5] = (-10, 15)
CLIPS_BY_LAYER[6] = (-10, 16)
CLIPS_BY_LAYER[7] = (-10, 16)
CLIPS_BY_LAYER[8] = (-10, 16)
CLIPS_BY_LAYER[9] = (-10, 16)
CLIPS_BY_LAYER[10] = (-10, 16)
CLIPS_BY_LAYER[11] = (-10, 16)
CLIPS_BY_LAYER[12] = (-10, 16)
CLIPS_BY_LAYER[13] = (-9, 16)
CLIPS_BY_LAYER[14] = (-9, 16)
CLIPS_BY_LAYER[15] = (-9, 15)
CLIPS_BY_LAYER[16] = (-9, 15)
CLIPS_BY_LAYER[17] = (-8, 13)
CLIPS_BY_LAYER[18] = (-8, 13)
CLIPS_BY_LAYER[19] = (-7, 12)
CLIPS_BY_LAYER[20] = (-7, 12)
CLIPS_BY_LAYER[21] = (-6, 11)
CLIPS_BY_LAYER[22] = (-6, 10)
CLIPS_BY_LAYER[23] = (-6, 10)
CLIPS_BY_LAYER[24] = (-0.35, 0.35)

SCALE_TOOLS_PATH = g.MODELS_DIR / f"scale_tools_{g.NAME}.joblib"

scale_tools = {}
is_scale_tools_loaded = os.path.exists(SCALE_TOOLS_PATH)

if is_scale_tools_loaded:
    print("Loading scale tools...")
    scale_tools = joblib.load(SCALE_TOOLS_PATH)
else:
    print("Generating scale tools...")
    sample_loaded = False
    for data_path in g.iter_data_paths(2, g.DataSetType.train):
        data = g.load_data(data_path)
        channel_dim = data.iloc[0]["data"].shape[-1]
        break

    clip_min = np.empty(channel_dim, dtype=np.float32)
    clip_max = np.empty(channel_dim, dtype=np.float32)

    for f in range(channel_dim):
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

    print("Clipping ranges per channel_dim:")
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

print("\nScaling...")
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
