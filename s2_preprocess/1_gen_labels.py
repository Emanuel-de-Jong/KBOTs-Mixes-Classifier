import joblib
import random
import csv
import s0_utils.global_params as g

MIN_MAX_SONGS_MULTIPLIER = 4
# Only for testing! -1 to disable.
TEST_LABEL_COUNT = -1

playlist_counts = {}
for folder in g.TRAIN_PLAYLISTS_DIR.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.rglob("*.mp3")))
        playlist_counts[folder.name] = mp3_count
for folder in g.TRAIN_PUBLIC_PLAYLISTS_DIR.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.rglob("*.mp3")))
        playlist_counts[folder.name] += mp3_count

sorted_playlist_counts = sorted(playlist_counts.items(), key=lambda x: x[1])
for name, count in sorted_playlist_counts:
    print(f"{name}: {count}")

min_song_count = max(min(playlist_counts.values()), 1)
max_song_count = int(round(float(min_song_count) * MIN_MAX_SONGS_MULTIPLIER))
print(f"\nMax song count: {max_song_count}")

labels = sorted([folder.name for folder in g.TRAIN_PLAYLISTS_DIR.iterdir() if folder.is_dir()])
if TEST_LABEL_COUNT != -1:
    labels = labels[:TEST_LABEL_COUNT]

joblib.dump(labels, g.MODELS_DIR / f"labels_{g.NAME}.joblib")

label_to_num = {label: i for i, label in enumerate(labels)}

def get_song_labels(data_set_type):
    with open(g.CACHE_DIR / f"labels_{data_set_type.name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song", "label", "is_public"])

        dir = g.TRAIN_PLAYLISTS_DIR if data_set_type == g.DataSetType.train else g.TEST_DIR
        playlist_count = 0
        for playlist_dir in dir.iterdir():
            if playlist_dir.is_dir():
                playlist_count += 1
                if TEST_LABEL_COUNT != -1 and playlist_count > TEST_LABEL_COUNT:
                    break

                songs = list(playlist_dir.rglob("*.mp3"))
                random.shuffle(songs)

                if data_set_type == g.DataSetType.train:
                    public_playlist_dir = g.TRAIN_PUBLIC_PLAYLISTS_DIR / playlist_dir.name
                    if public_playlist_dir.exists():
                        public_songs = list(public_playlist_dir.rglob("*.mp3"))
                        random.shuffle(public_songs)
                        songs.extend(public_songs)
                
                added_songs = 0
                for song_path in songs:
                    song = str(song_path.resolve())
                    label = label_to_num[playlist_dir.name]
                    is_public = True if song_path.is_relative_to(g.TRAIN_PUBLIC_PLAYLISTS_DIR) else False
                    w.writerow([song, label, is_public])
                    
                    added_songs += 1
                    if added_songs >= max_song_count:
                        break

get_song_labels(g.DataSetType.train)
get_song_labels(g.DataSetType.test)
