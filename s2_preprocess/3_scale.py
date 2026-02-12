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
    print("Loading scale tools...")
    scale_tools = joblib.load(SCALE_TOOLS_PATH)
else:
    print("Generating scale tools...")
    sample_loaded = False
    for data_path in g.iter_data_paths(2, g.DataSetType.train):
        data = g.load_data(data_path)
        layer_dim = data.iloc[0]["data"].shape[1]
        break

    clip_min = np.empty(layer_dim, dtype=np.float32)
    clip_max = np.empty(layer_dim, dtype=np.float32)

    for layer_idx in range(layer_dim):
        values = []

        for data_path in g.iter_data_paths(2, g.DataSetType.train):
            data = g.load_data(data_path)

            layer_vals = np.concatenate(
                [arr[:, layer_idx, :].reshape(-1) for arr in data["data"]],
                axis=0
            )
            values.append(layer_vals)

        values = np.concatenate(values, axis=0)
        clip_min[layer_idx] = np.percentile(values, 1)
        clip_max[layer_idx] = np.percentile(values, 99)

        del values
        gc.collect()

    scale_tools = {
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    print("Clipping ranges per layer:")
    print(pd.DataFrame({
        "clip_min": clip_min,
        "clip_max": clip_max
    }))

    joblib.dump(scale_tools, SCALE_TOOLS_PATH)

print("\nScaling...")
range_vals = scale_tools["clip_max"] - scale_tools["clip_min"]
range_vals[range_vals == 0] = 1.0

for data_set_type in g.DataSetType:
    out_idx = 0
    out_rows = []

    for data_path in g.iter_data_paths(2, data_set_type):
        data = g.load_data(data_path)

        for _, row in data.iterrows():
            arr = row["data"]

            arr = np.clip(
                arr,
                scale_tools["clip_min"][None, :, None],
                scale_tools["clip_max"][None, :, None]
            )

            arr = (arr - scale_tools["clip_min"][None, :, None]) / range_vals[None, :, None]
            arr = arr * 2.0 - 1.0

            row["data"] = arr.astype(np.float32, copy=False)

            out_rows.append(row)

            if len(out_rows) >= g.DATA_BATCH_SIZE:
                g.save_data(pd.DataFrame(out_rows), 3, data_set_type, out_idx)
                out_idx += 1
                out_rows = []

        gc.collect()

    if out_rows:
        g.save_data(pd.DataFrame(out_rows), 3, data_set_type, out_idx)
