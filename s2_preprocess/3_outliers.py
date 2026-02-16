import pandas as pd
import numpy as np
import joblib
import sys
import gc
import s0_utils.global_params as g

STEP = 3

MAX_REMOVE_PERC = 0.02
MIN_SONG_ROWS = 50
Z_SCORE_TRES = 1.2

IS_DUMMY_RUN = False

g.DATA_BATCH_SIZE = 10_000

# SKIPPING OUTLIERS FOR NOW
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
    
    if len(out_rows) > 0:
        g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)

sys.exit(0)

labels = joblib.load(g.MODELS_DIR / f"labels_{g.NAME}.joblib")
label_nums = list(range(len(labels)))
label_file_map = {label_value: [] for label_value in label_nums}

data_paths = list(g.iter_data_paths(STEP-1, g.DataSetType.train))
for file_idx, data_path in enumerate(data_paths):
    data = g.load_data(data_path)

    present_labels = data["label"].unique()
    for label_value in present_labels:
        if label_value in label_file_map:
            label_file_map[label_value].append(file_idx)

    del data
    gc.collect()

out_idx = 0
out_rows = []
for label_value in label_nums:
    file_indices = label_file_map[label_value]

    if len(file_indices) == 0:
        continue

    label_parts = []
    for file_idx in file_indices:
        data = g.load_data(data_paths[file_idx])
        label_data_part = data[data["label"] == label_value]
        if len(label_data_part) > 0:
            label_parts.append(label_data_part)
        del data

    label_data = pd.concat(label_parts, ignore_index=True)
    del label_parts
    gc.collect()

    first_arr = label_data["data"].iloc[0]
    tensor_shape = (len(label_data), first_arr.shape[1], first_arr.shape[2])
    row_tensors = np.empty(tensor_shape, dtype=np.float32)

    for tensor_idx, arr in enumerate(label_data["data"]):
        row_tensors[tensor_idx] = arr.reshape(arr.shape[1], arr.shape[2])

    if len(row_tensors) <= 2:
        for row in label_data.itertuples(index=False):
            out_rows.append(pd.Series(row._asdict()))
            if len(out_rows) >= g.DATA_BATCH_SIZE:
                if not IS_DUMMY_RUN:
                    g.save_data(pd.DataFrame(out_rows), STEP, g.DataSetType.train, out_idx)
                out_idx += 1
                out_rows = []
        del label_data, row_tensors
        gc.collect()
        continue

    layer_vars = []
    for layer_idx in range(row_tensors.shape[1]):
        flat = row_tensors[:, layer_idx, :].reshape(row_tensors.shape[0], -1)
        layer_vars.append(np.var(flat, axis=0).mean())

    layer_vars = np.asarray(layer_vars, dtype=np.float32)
    layer_weights = layer_vars / (layer_vars.sum() + 1e-12)

    centroid = np.median(row_tensors, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12)

    row_norms = row_tensors / (np.linalg.norm(row_tensors, axis=2, keepdims=True) + 1e-12)
    cos_sim = np.sum(row_norms * centroid_norm[None, :, :], axis=2)
    distances = np.sum(layer_weights[None, :] * (1.0 - cos_sim), axis=1).astype(np.float32)

    med = np.median(distances)
    mad = np.median(np.abs(distances - med))
    if mad < 1e-12:
        scale = np.std(distances) + 1e-12
    else:
        scale = mad * 1.4826

    robust_z = (distances - med) / scale

    remove_cap = int(np.floor(len(label_data) * MAX_REMOVE_PERC))

    keep_mask = np.ones(len(label_data), dtype=bool)

    removed_count = 0
    if remove_cap > 0:
        song_counts = label_data["song"].value_counts().to_dict()
        sorted_local_index = np.argsort(-robust_z)
        for local_pos in sorted_local_index:
            if robust_z[local_pos] < Z_SCORE_TRES:
                break
            if removed_count >= remove_cap:
                break

            song_name = label_data.at[local_pos, "song"]

            if song_counts[song_name] <= MIN_SONG_ROWS:
                continue

            keep_mask[local_pos] = False
            song_counts[song_name] -= 1
            removed_count += 1

    if removed_count > 0:
        log = f"{label_value}: Removed {removed_count} rows."
        if removed_count >= remove_cap:
            log += " Remove cap reached!"
        
        print(log)

    filtered_label_data = label_data.loc[keep_mask]
    for row in filtered_label_data.itertuples(index=False):
        out_rows.append(pd.Series(row._asdict()))
        if len(out_rows) >= g.DATA_BATCH_SIZE:
            if not IS_DUMMY_RUN:
                g.save_data(pd.DataFrame(out_rows), STEP, g.DataSetType.train, out_idx)
            out_idx += 1
            out_rows = []

    del label_data, filtered_label_data, row_tensors
    del layer_vars, layer_weights, centroid, centroid_norm
    del distances, robust_z, keep_mask
    gc.collect()

if len(out_rows) > 0:
    if not IS_DUMMY_RUN:
        g.save_data(pd.DataFrame(out_rows), STEP, g.DataSetType.train, out_idx)
