import pandas as pd
import numpy as np
import joblib
import gc
import os
import global_params as g
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

class SamplingType(Enum):
    none = 0
    undersample = 1
    oversample = 2

SCALE_TOOLS_PATH = g.CACHE_DIR / "scale_tools.joblib"
SCALE_BATCH_SIZE = 1000

VALIDATE_PERC = 0.2

SAMPLING = SamplingType.oversample
# -1 means no treshold
UNDERSAMPLE_TRES = 150
# -1 means no treshold
OVERSAMPLE_TRES = 250
OVERSAMPLE_COMPENSATION = int(OVERSAMPLE_TRES * 0.0)

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
    joblib.dump(scale_tools, g.CACHE_DIR / "scale_tools.joblib")

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data['label'].value_counts()

all_new_rows = []
validate_target = label_counts.max() * VALIDATE_PERC
for label in range(g.label_count):
    label_train_data = train_data[train_data["label"] == label]
    songs = label_train_data['song'].unique()
    np.random.shuffle(songs)

    total_rows = 0
    validate_songs = []
    organic_validate_target = int(round(VALIDATE_PERC * len(label_train_data)))
    for song in songs:
        song_rows = len(label_train_data[label_train_data['song'] == song])
        if total_rows + song_rows <= organic_validate_target:
            validate_songs.append(song)
            total_rows += song_rows

            if total_rows == organic_validate_target:
                break

    label_validate_idxs = label_train_data[label_train_data['song'].isin(validate_songs)].index
    g.data.loc[label_validate_idxs, "data_set"] = g.DataSetType.validate

    remaining_validate_target = int(validate_target - total_rows)
    if remaining_validate_target > 0:
        label_validate_data = g.data[(g.data["data_set"] == g.DataSetType.validate) & (g.data["label"] == label)]

        song_sizes = label_validate_data.groupby('song').size().sort_values()
        repeated_songs = np.tile(song_sizes.index.values, (remaining_validate_target // len(song_sizes)) + 1)

        new_rows = []
        total_dup_rows = 0
        for song in repeated_songs:
            song_rows = g.data[(g.data["song"] == song) & (g.data["label"] == label) & (g.data["data_set"] == g.DataSetType.validate)]

            if total_dup_rows + len(song_rows) >= remaining_validate_target:
                new_rows.append(song_rows[:remaining_validate_target - total_dup_rows])
                break

            new_rows.append(song_rows)
            total_dup_rows += len(song_rows)
        
        if new_rows:
            new_rows = pd.concat(new_rows).copy()
            all_new_rows.append(new_rows)

if all_new_rows:
    g.data = pd.concat([g.data] + all_new_rows, ignore_index=False)

validate_data = g.data[g.data["data_set"] == g.DataSetType.validate]
print("\n== Validate label counts ==")
for label, count in validate_data["label"].value_counts().items():
    print(f"{g.labels[label]}: {count}")

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data['label'].value_counts()

def undersample(label, sample_target):
    train_data = g.data[g.data["data_set"] == g.DataSetType.train]
    label_data = train_data[train_data['label'] == label]
    
    x = len(label_data) - sample_target

    song_counts = label_data['song'].value_counts().to_dict()
    last_removed = {s: label_data[label_data['song'] == s].index[-1] for s in song_counts}
    
    remove_idxs = []
    for _ in range(x):
        max_count = max(song_counts.values())
        candidates = [s for s, c in song_counts.items() if c == max_count]
        song = candidates[0]

        song_rows = label_data[label_data['song'] == song]
        idxs = song_rows.index.tolist()
        last_idx = last_removed[song]
        next_idx = idxs[(idxs.index(last_idx) - 1) % len(idxs)]
        
        remove_idxs.append(next_idx)
        last_removed[song] = next_idx
        song_counts[song] -= 1

    keep_idxs = label_data.index.difference(remove_idxs)
    new_train_data = pd.concat([
        train_data.loc[train_data['label'] != label],
        train_data.loc[keep_idxs]
    ])
    non_train_data = g.data[g.data["data_set"] != g.DataSetType.train]
    g.data = pd.concat([new_train_data, non_train_data], ignore_index=False)

def oversample(label, sample_target):
    train_data = g.data[g.data["data_set"] == g.DataSetType.train]
    label_data = train_data[train_data['label'] == label]
    
    x = sample_target - len(label_data)

    song_counts = label_data['song'].value_counts().to_dict()
    last_used = {s: label_data[label_data['song'] == s].index[0] for s in song_counts}
    
    new_rows = []
    for _ in range(x):
        min_count = min(song_counts.values())
        candidates = [s for s, c in song_counts.items() if c == min_count]
        song = candidates[0]

        song_rows = label_data[label_data['song'] == song]
        idxs = song_rows.index.tolist()
        last_idx = last_used[song]
        next_idx = idxs[(idxs.index(last_idx) + 1) % len(idxs)]
        
        new_rows.append(g.data.loc[next_idx].copy())
        last_used[song] = next_idx
        song_counts[song] += 1

    non_train_data = g.data[g.data["data_set"] != g.DataSetType.train]
    new_train_data = pd.concat([train_data, pd.DataFrame(new_rows)], ignore_index=False)
    g.data = pd.concat([new_train_data, non_train_data], ignore_index=False)

if SAMPLING == SamplingType.undersample:
    undersample_tres = UNDERSAMPLE_TRES if UNDERSAMPLE_TRES != -1 else label_counts.min()
    for label, count in label_counts.items():
        if count > undersample_tres:
            undersample(label, undersample_tres)
elif SAMPLING == SamplingType.oversample:
    oversample_tres = OVERSAMPLE_TRES if OVERSAMPLE_TRES != -1 else label_counts.max()
    for label, count in label_counts.items():
        if count > oversample_tres:
            undersample(label, oversample_tres)
        elif count < oversample_tres - OVERSAMPLE_COMPENSATION:
            oversample(label, oversample_tres - OVERSAMPLE_COMPENSATION)

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data["label"].value_counts()
print("\n== Train label counts after resample ==")
for label, count in label_counts.items():
    print(f"{g.labels[label]}: {count}")

g.save_data(4)
