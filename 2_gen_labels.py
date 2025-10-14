import random
import json
import csv
from pathlib import Path

MIN_PLAYLIST_SONGS = 9
MAX_PLAYLIST_SONGS = 15

music_dir = Path("music")
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

playlist_counts = []
for folder in music_dir.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.glob("*.mp3")))
        if mp3_count < MIN_PLAYLIST_SONGS:
            playlist_counts.append((folder.name, mp3_count))

playlist_counts.sort(key=lambda x: x[1])
if len(playlist_counts) > 0 and playlist_counts[0][1] < MIN_PLAYLIST_SONGS:
    for name, count in playlist_counts:
        print(f"{name}: {count}")

num_to_label = sorted([folder.name for folder in music_dir.iterdir() if folder.is_dir()])
label_to_num = {label: i for i, label in enumerate(num_to_label)}
with open(cache_dir / "num_to_label.json", "w") as f:
    json.dump(num_to_label, f, indent=4)
with open(cache_dir / "label_to_num.json", "w") as f:
    json.dump(label_to_num, f, indent=4)

with open(cache_dir / "labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath","label"])
    for playlist_dir in music_dir.iterdir():
        if playlist_dir.is_dir():
            songs = list(playlist_dir.glob("*.mp3"))
            random.shuffle(songs)
            
            added_songs = 0
            for p in songs:
                w.writerow([str(p.resolve()), label_to_num[playlist_dir.name]])
                
                added_songs += 1
                if added_songs >= MAX_PLAYLIST_SONGS:
                    break
            
            # if len(songs) < MIN_PLAYLIST_SONGS:
            #     needed = MIN_PLAYLIST_SONGS - len(songs)
            #     for i in range(needed):
            #         song = songs[i % len(songs)]
            #         w.writerow([str(song.resolve()), lbl_dir.name])
