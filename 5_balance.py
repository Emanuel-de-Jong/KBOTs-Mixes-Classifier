import pandas as pd
import numpy as np
import gc
import global_params as g

VALIDATE_PERC = 0.3
VALIDATE_MAX_NON_PUBLIC_PERC = 0.7

OVERSAMPLE_TRES_MULTIPLIER = 32 # 8*32=256
OVERSAMPLE_TRES = g.MIN_SONG_COUNT * OVERSAMPLE_TRES_MULTIPLIER

BATCH_SIZE = 10_000

def iter_data_files(step, data_set_type):
    count = g.get_data_count(step, data_set_type)
    for idx in range(count):
        yield idx

dfs = []
for idx in iter_data_files(4, g.DataSetType.train):
    g.load_data(4, g.DataSetType.train, idx)
    dfs.append(g.data)

train_data = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()

label_counts = train_data['label'].value_counts()

all_validate_rows = []
validate_target = label_counts.max() * VALIDATE_PERC
for label in range(g.LABEL_COUNT):
    label_train_data = train_data[train_data["label"] == label]

    non_public_data = label_train_data[label_train_data["is_public"] == False]
    songs = non_public_data['song'].unique()
    np.random.shuffle(songs)

    songs = songs[:int(len(songs) * VALIDATE_MAX_NON_PUBLIC_PERC)]

    public_data = label_train_data[label_train_data["is_public"] == True]
    public_songs = public_data['song'].unique()
    np.random.shuffle(public_songs)

    songs = np.concatenate([songs, public_songs])

    total_rows = 0
    validate_songs = []
    organic_validate_target = int(round(VALIDATE_PERC * len(label_train_data)))
    for song in songs:
        song_rows = label_train_data[label_train_data['song'] == song]
        if total_rows + len(song_rows) <= organic_validate_target:
            validate_songs.append(song)
            total_rows += len(song_rows)
            if total_rows == organic_validate_target:
                break

    label_validate_data = label_train_data[label_train_data['song'].isin(validate_songs)]
    all_validate_rows.append(label_validate_data)

    remaining_validate_target = int(validate_target - total_rows)
    if remaining_validate_target > 0 and not label_validate_data.empty:
        song_sizes = label_validate_data.groupby('song').size().sort_values()
        repeated_songs = np.tile(
            song_sizes.index.values,
            (remaining_validate_target // len(song_sizes)) + 1
        )

        new_rows = []
        total_dup_rows = 0
        for song in repeated_songs:
            song_rows = label_validate_data[label_validate_data["song"] == song]

            if total_dup_rows + len(song_rows) >= remaining_validate_target:
                new_rows.append(song_rows[:remaining_validate_target - total_dup_rows])
                break

            new_rows.append(song_rows)
            total_dup_rows += len(song_rows)

        if new_rows:
            all_validate_rows.append(pd.concat(new_rows))

validate_data = pd.concat(all_validate_rows, ignore_index=False)
print("\n== Validate label counts ==")
for label, count in validate_data["label"].value_counts().items():
    print(f"{g.LABELS[label]}: {count}")

for start in range(0, len(validate_data), BATCH_SIZE):
    g.data = validate_data.iloc[start:start + BATCH_SIZE]
    g.save_data(5, g.DataSetType.validate, start // BATCH_SIZE)

validate_idxs = validate_data.index.unique()

del validate_data
del all_validate_rows
gc.collect()

train_data = train_data.drop(validate_idxs)
label_counts = train_data['label'].value_counts()

def undersample(train_data, label, sample_target):
    label_data = train_data[train_data['label'] == label]
    x = len(label_data) - sample_target

    song_counts = label_data['song'].value_counts().to_dict()
    last_removed = {s: label_data[label_data['song'] == s].index[-1] for s in song_counts}

    songs_is_public = (
        label_data
        .groupby('song')['is_public']
        .first()
        .to_dict()
    )
    public_songs = {s for s, v in songs_is_public.items() if v}
    non_public_songs = set(song_counts) - public_songs

    remove_idxs = []
    for _ in range(x):
        active_songs = [s for s in public_songs if song_counts[s] > 0]
        if not active_songs:
            active_songs = [s for s in non_public_songs if song_counts[s] > 0]

        max_count = max(song_counts[s] for s in active_songs)
        song = [s for s in active_songs if song_counts[s] == max_count][0]

        song_rows = label_data[label_data['song'] == song]
        idxs = song_rows.index.tolist()
        last_idx = last_removed[song]
        next_idx = idxs[(idxs.index(last_idx) - 1) % len(idxs)]

        remove_idxs.append(next_idx)
        last_removed[song] = next_idx
        song_counts[song] -= 1

    return train_data.drop(remove_idxs)

def oversample(train_data, label, sample_target):
    label_data = train_data[train_data['label'] == label]
    x = sample_target - len(label_data)

    song_counts = label_data['song'].value_counts().to_dict()
    last_used = {s: label_data[label_data['song'] == s].index[0] for s in song_counts}

    new_rows = []
    for _ in range(x):
        min_count = min(song_counts.values())
        song = [s for s, c in song_counts.items() if c == min_count][0]

        song_rows = label_data[label_data['song'] == song]
        idxs = song_rows.index.tolist()
        last_idx = last_used[song]
        next_idx = idxs[(idxs.index(last_idx) + 1) % len(idxs)]

        new_rows.append(train_data.loc[next_idx].copy())
        last_used[song] = next_idx
        song_counts[song] += 1

    return pd.concat([train_data, pd.DataFrame(new_rows)], ignore_index=True)

train_sample_target = min(OVERSAMPLE_TRES, label_counts.max())
for label, count in label_counts.items():
    if count > train_sample_target:
        train_data = undersample(train_data, label, train_sample_target)
    elif count < train_sample_target:
        train_data = oversample(train_data, label, train_sample_target)

print("\n== Train label counts after resample ==")
for label, count in train_data["label"].value_counts().items():
    print(f"{g.LABELS[label]}: {count}")

for start in range(0, len(train_data), BATCH_SIZE):
    g.data = train_data.iloc[start:start + BATCH_SIZE]
    g.save_data(5, g.DataSetType.train, start // BATCH_SIZE)
