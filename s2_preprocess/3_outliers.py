import pandas as pd
import numpy as np
import joblib
import gc
import sys
import s0_utils.global_params as g

STEP = 3
MAX_REMOVE_PERC = 0.1
MIN_SONG_ROWS = 10

g.DATA_BATCH_SIZE = 3_000

for data_set_type in g.DataSetType:
    out_idx = 0
    out_rows = []
    for data_path in g.iter_data_paths(STEP-1, data_set_type):
        data = g.load_data(data_path)

        for _, row in data.iterrows():
            out_rows.append(row)
            if len(out_rows) >= g.DATA_BATCH_SIZE:
                g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)
                out_idx += 1
                out_rows = []

sys.exit(0)

labels = joblib.load(g.MODELS_DIR / f"labels_{g.NAME}.joblib")
label_nums = list(range(len(labels)))

for data_set_type in g.DataSetType:
    out_idx = 0
    out_rows = []

    for label_value in label_nums:
        label_parts = []

        for data_path in g.iter_data_paths(STEP-1, data_set_type):
            data = g.load_data(data_path)
            label_data = data[data["label"] == label_value]
            if len(label_data) > 0:
                label_parts.append(label_data)

        if len(label_parts) == 0:
            continue

        label_data = pd.concat(label_parts, ignore_index=True)
        del label_parts
        gc.collect()

        row_vectors = []
        for arr in label_data["data"]:
            row_vec = arr.mean(axis=0).reshape(-1).astype(np.float32, copy=False)
            row_vectors.append(row_vec)

        if len(row_vectors) <= 1:
            for _, row in label_data.iterrows():
                out_rows.append(row)
                if len(out_rows) >= g.DATA_BATCH_SIZE:
                    g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)
                    out_idx += 1
                    out_rows = []
            del label_data, row_vectors
            gc.collect()
            continue

        row_vectors = np.stack(row_vectors, axis=0)

        median_vec = np.median(row_vectors, axis=0)
        abs_dev = np.abs(row_vectors - median_vec)
        mad_vec = np.median(abs_dev, axis=0)
        mad_vec = np.where(mad_vec < 1e-8, 1e-8, mad_vec)

        robust_z = abs_dev / mad_vec
        row_scores = robust_z.mean(axis=1)

        remove_cap = int(np.floor(len(label_data) * MAX_REMOVE_PERC))

        keep_mask = np.ones(len(label_data), dtype=bool)

        if remove_cap > 0:
            song_counts = label_data["song"].value_counts().to_dict()
            sorted_local_index = np.argsort(-row_scores)

            removed_count = 0
            for local_pos in sorted_local_index:
                if removed_count >= remove_cap:
                    break

                song_name = label_data.at[local_pos, "song"]

                if song_counts[song_name] <= MIN_SONG_ROWS:
                    continue

                keep_mask[local_pos] = False
                song_counts[song_name] -= 1
                removed_count += 1

        filtered_label_data = label_data.loc[keep_mask]

        for _, row in filtered_label_data.iterrows():
            out_rows.append(row)
            if len(out_rows) >= g.DATA_BATCH_SIZE:
                g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)
                out_idx += 1
                out_rows = []

        del label_data, filtered_label_data, row_vectors, median_vec, abs_dev, mad_vec, robust_z, row_scores, keep_mask
        gc.collect()

    if len(out_rows) > 0:
        g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)

    del out_rows
    gc.collect()
