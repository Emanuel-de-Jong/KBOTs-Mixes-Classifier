import unidecode
import csv
import sys
import re
import os
from pathlib import Path

MIN_PLAYLIST_SONGS = 10

music_dir = Path("music")

# Clean folder names
for folder in os.listdir(music_dir):
    folder_path = music_dir / folder
    if folder_path.is_dir():
        new_name = folder
        if new_name.lower().startswith("kbot's "):
            new_name = new_name[7:]
        if new_name.lower().endswith(" mix"):
            new_name = new_name[:-4]
        new_name = new_name.strip()
        new_path = music_dir / new_name
        if new_path != folder_path:
            folder_path.rename(new_path)

# Clean mp3 names
for p in music_dir.glob("*/*.mp3"):
    old_stem = p.stem
    new_stem = unidecode.unidecode(old_stem)
    new_stem = re.sub(r'[^a-zA-Z0-9\s\.\-\_\,]', '', new_stem)
    new_stem = re.sub(r'\s+', ' ', new_stem).strip()
    
    if new_stem and new_stem != old_stem:
        new_filename = new_stem + p.suffix
        new_path = p.with_name(new_filename)
        
        if not new_path.exists():
            p.rename(new_path)

# Generate label file
playlist_counts = []
for folder in music_dir.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.glob("*.mp3")))
        if mp3_count < MIN_PLAYLIST_SONGS:
            playlist_counts.append((folder.name, mp3_count))

playlist_counts.sort(key=lambda x: x[1])
# if playlist_counts[0][1] < 10:
#     for name, count in playlist_counts:
#         print(f"{name}: {count}")

with open("labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath","label"])
    for lbl_dir in music_dir.iterdir():
        if lbl_dir.is_dir():
            songs = list(lbl_dir.glob("*.mp3"))
            i = 0
            for p in songs:
                i += 1
                # if i > 3:
                #     break

                w.writerow([str(p.resolve()), lbl_dir.name])
            
            # if len(songs) < MIN_PLAYLIST_SONGS:
            #     needed = MIN_PLAYLIST_SONGS - len(songs)
            #     for i in range(needed):
            #         song = songs[i % len(songs)]
            #         w.writerow([str(song.resolve()), lbl_dir.name])
