import re
import sys
import yaml
import math
import random
import shutil
import s0_utils.global_params as g
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse

SONGS_PER_GENRE = 50

STEP_DIR = Path("s1_prep")

def get_playlist_name(url):
    url_obj = urlparse(url)
    unsanitized_playlist_name = url_obj.path + url_obj.query
    playlist_name = re.sub(r"[^a-zA-Z0-9]", "", unsanitized_playlist_name)
    return playlist_name[:32]

if any(f.is_dir() for f in g.TRAIN_PUBLIC_PLAYLISTS_DIR.iterdir()):
    sys.exit(0)

with open(STEP_DIR / "public_playlists.yaml", "r") as f:
    categories_playlists = yaml.safe_load(f)
    categories_playlists.pop("requirements", None)

for category, genres_playlists in categories_playlists.items():
    for genre, genre_playlists in tqdm(
            genres_playlists.items(),
            desc="Genres",
            position=0):
        valid_playlists = [p for p in genre_playlists if p]
        if not valid_playlists:
            continue

        genre_dir = g.DLS_DIR / genre

        songs_by_playlist = {}
        for playlist_url in valid_playlists:
            playlist_dir = genre_dir / get_playlist_name(playlist_url)
            if not playlist_dir.exists():
                continue

            songs = list(playlist_dir.rglob("*.mp3"))
            if len(songs) == 0:
                continue

            songs_by_playlist[playlist_dir] = songs
        
        playlist_count = len(songs_by_playlist.keys())
        if playlist_count == 0:
            continue

        song_target = math.ceil(SONGS_PER_GENRE / playlist_count)

        song_target_by_playlist = {}
        for playlist_dir, songs in songs_by_playlist.items():
            song_target_by_playlist[playlist_dir] = min(len(songs), song_target)
        
        song_count_to_balance = SONGS_PER_GENRE - sum(song_target_by_playlist.values())
        while song_count_to_balance > 0:
            has_progress = False
            for playlist_dir, songs in songs_by_playlist.items():
                if len(songs) > song_target_by_playlist[playlist_dir]:
                    has_progress = True
                    song_target_by_playlist[playlist_dir] += 1
                    song_count_to_balance -= 1
            
            if not has_progress:
                break
        
        for playlist_dir, songs in songs_by_playlist.items():
            random.shuffle(songs)

            song_target = song_target_by_playlist[playlist_dir]
            for i in range(song_target):
                target_path = songs[i]

                destination_dir = g.TRAIN_PUBLIC_PLAYLISTS_DIR / genre
                destination_dir.mkdir(exist_ok=True)

                destination_filename = f"{target_path.stem}_{random.randint(100, 999)}{target_path.suffix}"
                destination_path = destination_dir / destination_filename

                shutil.copy2(target_path, destination_path)
