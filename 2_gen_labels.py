import joblib
import random
import csv
import global_params as g

MIN_PLAYLIST_SONGS = 10
MAX_PLAYLIST_SONGS = 25
# Only for testing! -1 to disable.
TEST_LABEL_COUNT = -1

playlist_counts = []
for folder in g.TRAIN_DIR.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.glob("*.mp3")))
        if mp3_count < MIN_PLAYLIST_SONGS:
            playlist_counts.append((folder.name, mp3_count))

playlist_counts.sort(key=lambda x: x[1])
if len(playlist_counts) > 0 and playlist_counts[0][1] < MIN_PLAYLIST_SONGS:
    for name, count in playlist_counts:
        print(f"{name}: {count}")

labels = sorted([folder.name for folder in g.TRAIN_DIR.iterdir() if folder.is_dir()])
if TEST_LABEL_COUNT != -1:
    labels = labels[:TEST_LABEL_COUNT]

joblib.dump(labels, g.CACHE_DIR / f"labels_{g.NAME}.joblib")

label_to_num = {label: i for i, label in enumerate(labels)}

def get_song_labels(data_set_type):
    with open(g.CACHE_DIR / f"labels_{data_set_type.name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song","label"])

        dir = g.TRAIN_DIR if data_set_type == g.DataSetType.train else g.TEST_DIR
        playlist_count = 0
        for playlist_dir in dir.iterdir():
            if playlist_dir.is_dir():
                playlist_count += 1
                if TEST_LABEL_COUNT != -1 and playlist_count > TEST_LABEL_COUNT:
                    break

                songs = list(playlist_dir.glob("*.mp3"))
                random.shuffle(songs)
                
                added_songs = 0
                for p in songs:
                    w.writerow([str(p.resolve()), label_to_num[playlist_dir.name]])
                    
                    added_songs += 1
                    if added_songs >= MAX_PLAYLIST_SONGS:
                        break

get_song_labels(g.DataSetType.train)
get_song_labels(g.DataSetType.test)
