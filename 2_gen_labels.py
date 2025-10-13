import random
import csv
from pathlib import Path

MIN_PLAYLIST_SONGS = 9
MAX_PLAYLIST_SONGS = 10

music_dir = Path("music")

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

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
with open(cache_dir / "labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath","label"])
    for lbl_dir in music_dir.iterdir():
        if lbl_dir.is_dir():
            songs = list(lbl_dir.glob("*.mp3"))
            random.shuffle(songs)
            
            added_songs = 0
            for p in songs:
                w.writerow([str(p.resolve()), lbl_dir.name])
                
                added_songs += 1
                if added_songs >= MAX_PLAYLIST_SONGS:
                    break
            
            # if len(songs) < MIN_PLAYLIST_SONGS:
            #     needed = MIN_PLAYLIST_SONGS - len(songs)
            #     for i in range(needed):
            #         song = songs[i % len(songs)]
            #         w.writerow([str(song.resolve()), lbl_dir.name])
